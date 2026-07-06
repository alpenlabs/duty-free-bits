//! The `a·x + b` protocol driver: composes the four straight-line steps —
//! repeated one-hot scaling + free recombination — that evaluate the affine
//! maps over a primorial ring.
//!
//! Given an n-bit input x and coefficients (a, b) reduced mod each CRT prime,
//! [`build_s_aff`] computes a·x + b in Z_M (where M = Π p_i) as four
//! straight-line steps over bare labels — no gate graph, no worklist:
//!
//!   1. **Chunk conversion**: partition n bits into ⌈n/lg n⌉ chunks, grow a
//!      one-hot over each and upcast it to a word in Z_{2^ℓ}
//!      ([`crate::gc::chunk`]).
//!   2. **Extract**: per prime, free-recombine the words into r_i ≡ x (mod
//!      p_i), then decompose it (one-hot scaling + upcast) into sub-chunk bits
//!      + a one-hot ([`crate::gc::extract`]).
//!   3. **Fold**: reduce to a length-p_i one-hot `h_p` of x mod p_i
//!      ([`crate::gc::fold`]).
//!   4. **Body**: the information-theoretic GC's one-hot scaling delivers
//!      a·(x mod p_i) + b from `h_p` ([`crate::gc::body`]).

use crate::crt::{CrtParams, pow2_mod};
use crate::crypto::nonce;
use crate::gc::body::{body_batch_eval, body_batch_garble};
use crate::gc::chunk::{chunk_batch_eval, chunk_batch_garble, chunk_nonces};
use crate::gc::extract::{
    compute_sub_widths, extract_batch_eval, extract_batch_garble, extract_nonces,
};
use crate::gc::fold::{fold_batch_eval, fold_batch_garble};
use crate::label::{self, LAMBDA, Label};
use rand::{Rng, RngExt};

/// Maximum sub-chunk width for the sub-chunk extraction optimization.
/// 2^8 = 256 one-hot positions per sub-chunk.
const MAX_SUB_CHUNK_WIDTH: u32 = 8;

/// S-batch size for the residue body: how many of the `S` affine maps one
/// body call handles. 128 fills a full λ-bit CCRH block (so packing isn't
/// wasted) while keeping the per-batch working set small.
const RESIDUE_BATCH_SIZE: usize = 128;

/// First bulk-domain CCRH id available to the fold/body steps; the
/// chunk/extract tree hashes allocate their windows above it. Reserving a
/// `2^32` floor keeps the space open for future bulk consumers.
const BULK_NONCE_FLOOR: u64 = 1 << 32;

/// Sample a uniform mask (a garbler's share) in Z_modulus (modulus must be a power of two).
fn sample_cf_mask<R: Rng>(rng: &mut R, modulus: u64) -> Label {
    assert!(modulus.is_power_of_two());
    let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
    Label::from_coords(&coords, modulus)
}

/// Run `f`, adding its wall time to `*secs` and its CCRH-block delta to
/// `*blocks` (the latter nonzero only under the `count-hashes` feature). Wraps
/// each garble/eval step so the driver reads as the pipeline, not the timers.
fn measure<T>(secs: &mut f64, blocks: &mut u64, f: impl FnOnce() -> T) -> T {
    let b0 = crate::crypto::hash_blocks();
    let t = std::time::Instant::now();
    let out = f();
    *secs += t.elapsed().as_secs_f64();
    *blocks += crate::crypto::hash_blocks() - b0;
    out
}

/// Per-stage telemetry of one [`build_s_aff`] run.
///
/// Times are wall-clock seconds; `*_hash_blocks` are CCRH AES blocks (nonzero
/// only under the `count-hashes` feature); the ledger fields count communication + hashes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Chunk-step garble wall time.
    pub chunk_garble_secs: f64,
    /// Chunk-step eval wall time.
    pub chunk_eval_secs: f64,
    /// Extract-step garble wall time.
    pub extract_garble_secs: f64,
    /// Extract-step eval wall time.
    pub extract_eval_secs: f64,
    /// Fold-step garble wall time.
    pub fold_garble_secs: f64,
    /// Fold-step eval wall time.
    pub fold_eval_secs: f64,
    /// Body-step garble wall time.
    pub body_garble_secs: f64,
    /// Body-step eval wall time.
    pub body_eval_secs: f64,
    /// Garbler-emitted material in bits (the scaling ciphertexts).
    pub program_bits: usize,
    /// Scaling width of the boolean-label steps (1–3), in `lg|G|` units (each pays λ bits).
    pub join_complexity_cf: usize,
    /// Scaling width of the `Z_p`-share body (4), in bits.
    pub join_complexity_ncf: usize,
    /// Ledger CCRH blocks from the boolean-label steps' scalings.
    pub hash_count_cf: usize,
    /// Ledger CCRH blocks from the `Z_p`-share body's scalings (bulk-pack rebated).
    pub hash_count_ncf: usize,
    /// Measured garbler CCRH blocks (`count-hashes` feature; else 0).
    pub garble_hash_blocks: u64,
    /// Measured evaluator CCRH blocks (`count-hashes` feature; else 0).
    pub eval_hash_blocks: u64,
}

impl Stats {
    /// Total garble/eval wall time.
    pub fn garble_secs(&self) -> f64 {
        self.chunk_garble_secs
            + self.extract_garble_secs
            + self.fold_garble_secs
            + self.body_garble_secs
    }
    /// See [`garble_secs`](Self::garble_secs).
    pub fn eval_secs(&self) -> f64 {
        self.chunk_eval_secs + self.extract_eval_secs + self.fold_eval_secs + self.body_eval_secs
    }

    /// Fold a boolean-label step's cost (chunk/extract/fold) into the ledger.
    fn add_cf(&mut self, c: crate::gc::Cost) {
        self.program_bits += c.program_bits;
        self.join_complexity_cf += c.join_complexity;
        self.hash_count_cf += c.hash_count;
    }

    /// Fold the `Z_p`-share body's cost into the ledger.
    fn add_ncf(&mut self, c: crate::gc::Cost) {
        self.program_bits += c.program_bits;
        self.join_complexity_ncf += c.join_complexity;
        self.hash_count_ncf += c.hash_count;
    }
}

/// Garble + evaluate the affine maps as four straight-line steps: chunk
/// conversion → per-prime extract → fold → body.
///
/// The steps are [`crate::gc::chunk`], [`crate::gc::extract`],
/// [`crate::gc::fold`], [`crate::gc::body`]. No gate graph, no worklist: the
/// evaluator holds `x_bits` in the clear, so it knows every active position and
/// every derivation order is closed-form.
///
/// One parameter restriction: the extract step represents width-`l` arithmetic
/// labels as u32 lanes, so every sub-chunk width must be in `2..=31`; shapes
/// with a width-1 trailing sub-chunk (`ell ≡ 1 mod 8`, e.g. n = 1 or n = 26..=32
/// with the 80-prime set) are rejected loudly.
pub fn build_s_aff<R: rand::Rng>(
    rng: &mut R,
    x_bits: &[u64],
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> (Vec<Vec<u64>>, Stats) {
    let n = params.n as usize;
    assert_eq!(x_bits.len(), n);
    for &b in x_bits {
        assert!(b < 2, "x_bits entries must be 0 or 1");
    }
    assert_eq!(a_residues.len(), params.num_primes);
    assert_eq!(b_residues.len(), params.num_primes);
    let s_dim = a_residues[0].len();
    for i in 0..params.num_primes {
        assert_eq!(a_residues[i].len(), s_dim);
        assert_eq!(b_residues[i].len(), s_dim);
    }

    let ell = params.ell;
    let chunk_size = params.chunk_size as usize;
    let work_mod = 1u64 << ell;
    let num_chunks = params.num_chunks;
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);
    let first_width = sub_widths[0];

    let delta: u128 = rng.random::<u128>() | 1;
    let d2 = label::delta_r(delta, 2);
    let mut stats = Stats::default();

    // ---- Nonce windows (paper App. A, Def. 4) ----
    let nonce_layout = NonceLayout::new(params, s_dim, &sub_widths);
    let chunk_solo_ids = nonce_layout.chunk_solo_ids;
    let chunk_bulk_ids = nonce_layout.chunk_bulk_ids;
    let ex_solo_ids = nonce_layout.ex_solo_ids;
    let ex_bulk_ids = nonce_layout.ex_bulk_ids;
    let solo_chunk_base = nonce_layout.solo_chunk_base;
    let solo_extract_base = nonce_layout.solo_extract_base;
    let num_batches = nonce_layout.num_batches;
    let prime_nonce_bases = nonce_layout.prime_nonce_bases.clone();
    let bulk_chunk_base = nonce_layout.bulk_chunk_base;
    let bulk_extract_base = nonce_layout.bulk_extract_base;

    // ---- Input bits: masks sampled, labels = mask + bit·Δ₂. ----
    let bit_masks: Vec<Label> = (0..n).map(|_| sample_cf_mask(rng, 2)).collect();
    let bit_labels: Vec<Label> = bit_masks
        .iter()
        .zip(x_bits)
        .map(|(m, &b)| {
            if b == 1 {
                label::add(m, &d2)
            } else {
                m.clone()
            }
        })
        .collect();

    // ---- Stage 1: chunk conversion. ----
    let zero_bit_mask = Label::zero(2); // constant-0 padding for a short last chunk
    let mut chunk_word_masks: Vec<Label> = Vec::with_capacity(num_chunks);
    let mut chunk_word_labels: Vec<Label> = Vec::with_capacity(num_chunks);
    let chunk_values: Vec<u64> = (0..num_chunks)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            x_bits[start..end]
                .iter()
                .enumerate()
                .fold(0u64, |acc, (j, &b)| acc | (b << j))
        })
        .collect();
    // `c` indexes chunk_values and derives start/end/nonce bases — not a plain map.
    #[allow(clippy::needless_range_loop)]
    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let mut masks: Vec<Label> = bit_masks[start..end].to_vec();
        let mut labels: Vec<Label> = bit_labels[start..end].to_vec();
        while masks.len() < chunk_size {
            masks.push(zero_bit_mask.clone());
            labels.push(zero_bit_mask.clone());
        }
        let bulk = bulk_chunk_base + c as u64 * chunk_bulk_ids;
        let solo = solo_chunk_base + c as u64 * chunk_solo_ids;

        let g = measure(
            &mut stats.chunk_garble_secs,
            &mut stats.garble_hash_blocks,
            || chunk_batch_garble(&masks, ell, delta, bulk, solo),
        );

        let w = measure(
            &mut stats.chunk_eval_secs,
            &mut stats.eval_hash_blocks,
            || chunk_batch_eval(&labels, chunk_values[c], ell, &g.scale, &g.pin, bulk, solo),
        );

        stats.add_cf(g.cost);
        chunk_word_masks.push(g.word_mask);
        chunk_word_labels.push(w);
    }

    // ---- Stage 2..4 per prime: extract → fold → body. ----
    let mut all_outputs: Vec<Vec<u64>> = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        let coeffs: Vec<u64> = (0..num_chunks)
            .map(|c| pow2_mod((c * chunk_size) as u32, p_i))
            .collect();
        let r_value: u64 = chunk_values.iter().zip(&coeffs).map(|(&v, &c)| c * v).sum();
        assert!(
            r_value < work_mod,
            "r_i overflows 2^ell — CrtParams.ell undersized for these primes/n"
        );
        let hot_i = (r_value % p_i) as usize;

        let bulk = bulk_extract_base + i as u64 * ex_bulk_ids;
        let solo = solo_extract_base + i as u64 * ex_solo_ids;

        let g = measure(
            &mut stats.extract_garble_secs,
            &mut stats.garble_hash_blocks,
            || extract_batch_garble(&chunk_word_masks, &coeffs, &sub_widths, delta, bulk, solo),
        );

        let ex = measure(
            &mut stats.extract_eval_secs,
            &mut stats.eval_hash_blocks,
            || {
                extract_batch_eval(
                    &chunk_word_labels,
                    &coeffs,
                    r_value,
                    &sub_widths,
                    &g.diffs,
                    bulk,
                    solo,
                )
            },
        );

        stats.add_cf(g.cost);

        // ---- Fold step. ----
        let fold_nonce_base = prime_nonce_bases[i] + num_batches as u64 * p_i;
        let fold_g = measure(
            &mut stats.fold_garble_secs,
            &mut stats.garble_hash_blocks,
            || {
                fold_batch_garble(
                    p_i,
                    &g.first_bin_hot_masks,
                    &g.fold_bit_masks,
                    first_width,
                    fold_nonce_base,
                )
            },
        );

        let h_p_labels = measure(
            &mut stats.fold_eval_secs,
            &mut stats.eval_hash_blocks,
            || {
                fold_batch_eval(
                    p_i,
                    r_value,
                    &ex.first_bin_hot_labels,
                    &ex.fold_bit_labels,
                    &fold_g.join_diffs,
                    first_width,
                    fold_nonce_base,
                )
            },
        );
        stats.add_cf(fold_g.cost);
        let h_p_masks = fold_g.h_p_masks;

        // ---- Body batches. ----
        let weights: Vec<u64> = (0..p_i).collect();
        let mut prime_outputs: Vec<u64> = Vec::with_capacity(s_dim);
        let mut start = 0usize;
        while start < s_dim {
            let end = (start + RESIDUE_BATCH_SIZE).min(s_dim);
            let a_batch: Vec<u64> = a_residues[i][start..end].iter().map(|&a| a % p_i).collect();
            let b_batch: Vec<u64> = b_residues[i][start..end].iter().map(|&b| b % p_i).collect();
            let batch_idx = start / RESIDUE_BATCH_SIZE;
            let nonce_base = prime_nonce_bases[i] as usize + batch_idx * p_i as usize;

            let g_out = measure(
                &mut stats.body_garble_secs,
                &mut stats.garble_hash_blocks,
                || body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &weights, nonce_base),
            );

            let result_labels = measure(
                &mut stats.body_eval_secs,
                &mut stats.eval_hash_blocks,
                || {
                    body_batch_eval(
                        p_i,
                        hot_i,
                        &h_p_labels,
                        &g_out.join_diffs,
                        &b_batch,
                        &weights,
                        nonce_base,
                    )
                },
            );

            for (l, m) in result_labels.iter().zip(g_out.result_masks.iter()) {
                let value = if l >= m { l - m } else { l + p_i - m };
                prime_outputs.push(value);
            }
            stats.add_ncf(g_out.cost);
            start = end;
        }
        all_outputs.push(prime_outputs);
    }

    (all_outputs, stats)
}

/// The complete CCRH nonce-window layout (paper App. A, Def. 4: no two
/// garbler queries may share a (domain, id) — a fresh nonce per scaling
/// ciphertext).
///
/// Solo domain (the width-`l` upcast pads): chunk-step windows first, then the
/// per-prime extract windows. Bulk domain (the boolean one-hot growing hashes
/// and the fold/body steps): the fold/body per-prime windows from
/// [`BULK_NONCE_FLOOR`], then the chunk windows, then the per-prime extract
/// windows above those. This struct is the single source of truth the driver
/// draws its bases from; `test_nonce_windows_disjoint` pins the partition (a
/// regression output tests cannot catch — the test explains why).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct NonceLayout {
    pub chunk_solo_ids: u64,
    pub chunk_bulk_ids: u64,
    pub ex_solo_ids: u64,
    pub ex_bulk_ids: u64,
    pub solo_chunk_base: u64,
    pub solo_extract_base: u64,
    pub num_batches: usize,
    pub prime_nonce_bases: Vec<u64>,
    pub prime_window_sizes: Vec<u64>,
    pub bulk_chunk_base: u64,
    pub bulk_extract_base: u64,
    num_chunks: usize,
    num_primes: usize,
}

impl NonceLayout {
    pub(crate) fn new(params: &CrtParams, s_dim: usize, sub_widths: &[u32]) -> Self {
        let fold_bits: u32 = sub_widths[1..].iter().sum();
        let (chunk_bulk_ids, chunk_solo_ids) = chunk_nonces(params.chunk_size);
        let (ex_bulk_ids, ex_solo_ids) = extract_nonces(sub_widths);
        let solo_chunk_base = 0u64;
        let solo_extract_base = solo_chunk_base + params.num_chunks as u64 * chunk_solo_ids;
        let num_batches = s_dim.div_ceil(RESIDUE_BATCH_SIZE);
        let prime_window_sizes: Vec<u64> = params
            .primes
            .iter()
            .map(|&p| (num_batches as u64 + fold_bits as u64) * p)
            .collect();
        let prime_nonce_bases = nonce::windows(BULK_NONCE_FLOOR, &prime_window_sizes);
        let bulk_tree_base = prime_nonce_bases
            .last()
            .zip(prime_window_sizes.last())
            .map(|(&b, &s)| b + s)
            .unwrap_or(BULK_NONCE_FLOOR);
        let bulk_chunk_base = bulk_tree_base;
        let bulk_extract_base = bulk_chunk_base + params.num_chunks as u64 * chunk_bulk_ids;
        assert!(
            bulk_extract_base + params.num_primes as u64 * ex_bulk_ids < (1u64 << 63),
            "CCRH nonce space exhausted"
        );
        assert!(
            solo_extract_base + params.num_primes as u64 * ex_solo_ids < (1u64 << 63),
            "solo-domain nonce space exhausted"
        );
        NonceLayout {
            chunk_solo_ids,
            chunk_bulk_ids,
            ex_solo_ids,
            ex_bulk_ids,
            solo_chunk_base,
            solo_extract_base,
            num_batches,
            prime_nonce_bases,
            prime_window_sizes,
            bulk_chunk_base,
            bulk_extract_base,
            num_chunks: params.num_chunks,
            num_primes: params.num_primes,
        }
    }

    /// Every reserved window as `(is_bulk, base, len)` — the disjointness test
    /// walks this enumeration, so it covers exactly what the driver uses.
    /// The per-prime fold/body reservation is treated as one window (its
    /// internal body-batch/fold split is defined by the driver).
    #[cfg(test)]
    pub(crate) fn windows(&self) -> Vec<(bool, u64, u64)> {
        let mut w = Vec::new();
        for c in 0..self.num_chunks as u64 {
            w.push((
                false,
                self.solo_chunk_base + c * self.chunk_solo_ids,
                self.chunk_solo_ids,
            ));
            w.push((
                true,
                self.bulk_chunk_base + c * self.chunk_bulk_ids,
                self.chunk_bulk_ids,
            ));
        }
        for i in 0..self.num_primes {
            w.push((
                false,
                self.solo_extract_base + i as u64 * self.ex_solo_ids,
                self.ex_solo_ids,
            ));
            w.push((
                true,
                self.bulk_extract_base + i as u64 * self.ex_bulk_ids,
                self.ex_bulk_ids,
            ));
            w.push((true, self.prime_nonce_bases[i], self.prime_window_sizes[i]));
        }
        w
    }
}

#[cfg(test)]
mod nonce_layout_tests {
    use super::*;

    /// Def-4 freshness rests on this partition: colliding windows would still
    /// decode correctly (both parties share the layout), so no output test
    /// can catch a regression — this test is the guard. It pins, per domain,
    /// that every reserved window is pairwise disjoint, that bulk ids
    /// respect the `[0, 2^32)` reserved floor, and that everything
    /// stays below the bit-63 domain flag.
    #[test]
    fn test_nonce_windows_disjoint() {
        use crate::crt::bigint::FIRST_80_PRIMES;
        for (primes, n, s_dim) in [
            (&FIRST_80_PRIMES[..], 256u32, 1536usize),
            (&FIRST_80_PRIMES[..], 64, 257),
            (&FIRST_80_PRIMES[..], 33, 1),
            (&[2u64][..], 8, 129),
            (&[2u64, 3, 409][..], 16, 128),
            (&[409u64][..], 12, 5),
        ] {
            let params = CrtParams::from_primes(primes, n);
            let sub_widths = compute_sub_widths(params.ell, MAX_SUB_CHUNK_WIDTH);
            if sub_widths.iter().any(|&w| w < 2) {
                continue; // width-1 shapes are rejected
            }
            let nonce_layout = NonceLayout::new(&params, s_dim, &sub_widths);
            for bulk in [false, true] {
                let mut ws: Vec<(u64, u64)> = nonce_layout
                    .windows()
                    .iter()
                    .filter(|w| w.0 == bulk && w.2 > 0)
                    .map(|w| (w.1, w.2))
                    .collect();
                ws.sort_unstable();
                for pair in ws.windows(2) {
                    assert!(
                        pair[0].0 + pair[0].1 <= pair[1].0,
                        "overlapping {} windows {:?} / {:?} (primes={:?} n={n} s={s_dim})",
                        if bulk { "bulk" } else { "solo" },
                        pair[0],
                        pair[1],
                        primes
                    );
                }
                let (last_base, last_len) = *ws.last().unwrap();
                assert!(last_base + last_len < (1u64 << 63), "domain-flag overflow");
                if bulk {
                    assert!(
                        ws[0].0 >= BULK_NONCE_FLOOR,
                        "bulk ids must stay above the reserved floor"
                    );
                }
            }
        }
    }
}
