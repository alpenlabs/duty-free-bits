//! The affine-map protocol as a partial garbling scheme: `garble`, `encode`,
//! `eval` as three separable procedures.
//!
//! [`crate::affine::build_s_aff`] runs garbler and evaluator in one pass, which
//! is what the benchmarks want. A caller that composes this protocol with other
//! garbled objects (an outer scheme whose input labels are the affine maps'
//! outputs) needs the two halves apart: the garbler produces a [`Program`] and
//! keeps the [`EncodingInfo`]; the evaluator receives the program and, for her
//! bits, the [`InputLabels`], and decodes `a·x + b mod p_i` for every CRT prime.
//! The steps, nonce layout, and ciphertexts are exactly those of `build_s_aff`;
//! only the control flow is split.
//!
//! The evaluator holds `x` in the clear (the scheme is *partial*), so [`eval`]
//! takes `x_bits` alongside the labels: every active position is derived from it.
//!
//! # What the caller owns
//!
//! The protocol computes `a·x + b` over `Z_M` and returns its residues; it does
//! not reduce modulo a target prime `q`, and it does not hide the integer
//! `a·x + b`. A caller whose target is `a·x + b mod q` must smudge each offset
//! before deriving residues, `b' = b + μ·q` with `μ` uniform and wide enough that
//! `a·x + b'` stays below `M` while the quotient `⌊(a·x + b)/q⌋` is hidden to the
//! statistical distance it needs (Duty-Free Bits, Lemma 3.2 and Thm 5.2), and
//! then reduce the reconstructed integer modulo `q`. Passing raw residues of
//! `(a, b)` reveals `a` and `b` after reconstruction.

use crate::affine::{MAX_SUB_CHUNK_WIDTH, NonceLayout, RESIDUE_BATCH_SIZE};
use crate::crt::{CrtParams, pow2_mod};
use crate::gc::body::{body_batch_eval, body_batch_garble};
use crate::gc::chunk::{chunk_batch_eval, chunk_batch_garble};
use crate::gc::extract::{
    SubChunkDiffs, compute_sub_widths, extract_batch_eval, extract_batch_garble,
};
use crate::gc::fold::{fold_batch_eval, fold_batch_garble};
use crate::gc::onehot::{Wide, Z2};
use crate::hash::lg_modulus;
use crate::label::{self, LAMBDA, Label};
use rand::{Rng, RngExt, SeedableRng};

/// The scaling ciphertexts of one chunk step.
#[derive(Debug)]
struct ChunkCiphertexts {
    scale: Vec<Z2>,
    pin: Wide,
}

/// One body batch: the scaling residues and the garbler's output masks. The
/// masks are decoding material (`value = label − mask`); they are shares of
/// values the evaluator is entitled to, so sending them reveals nothing beyond
/// the outputs.
#[derive(Debug)]
struct BodyBatch {
    join_diffs: Vec<u64>,
    result_masks: Vec<u64>,
}

/// The per-prime ciphertexts: extract, fold, body.
#[derive(Debug)]
struct PrimeCiphertexts {
    extract: Vec<SubChunkDiffs>,
    fold: Vec<Label>,
    body: Vec<BodyBatch>,
}

/// The garbled program: everything the garbler sends besides the input labels.
#[derive(Debug)]
pub struct Program {
    params: CrtParams,
    s_dim: usize,
    chunks: Vec<ChunkCiphertexts>,
    primes: Vec<PrimeCiphertexts>,
    /// Ciphertext bits, as counted by [`crate::affine::Stats::program_bits`].
    program_bits: usize,
    /// Bits of the body's output masks (decoding material).
    mask_bits: usize,
}

impl Program {
    /// Number of affine maps `S`.
    pub fn s_dim(&self) -> usize {
        self.s_dim
    }
    /// The CRT parameters the program was garbled for.
    pub fn params(&self) -> &CrtParams {
        &self.params
    }
    /// Ciphertext bits (the scaling material), excluding output masks.
    pub fn program_bits(&self) -> usize {
        self.program_bits
    }
    /// Bits of the output masks the evaluator subtracts to decode.
    pub fn mask_bits(&self) -> usize {
        self.mask_bits
    }
}

/// The garbler's encoding information: one boolean mask per input bit and the
/// bit-domain offset `Δ₂`. The label of bit value `b` is `mask + b·Δ₂`.
#[derive(Debug)]
pub struct EncodingInfo {
    bit_masks: Vec<Label>,
    d2: Label,
}

impl EncodingInfo {
    /// Number of input bits `n`.
    pub fn n(&self) -> usize {
        self.bit_masks.len()
    }
}

/// The evaluator's input labels, one boolean label per bit.
#[derive(Debug, Clone)]
pub struct InputLabels {
    labels: Vec<Label>,
}

impl InputLabels {
    /// Number of input bits.
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    /// Whether there are no labels.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// A uniform boolean mask (a garbler's share in `Z_2`).
fn sample_bit_mask<R: Rng>(rng: &mut R) -> Label {
    let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..2u64)).collect();
    Label::from_coords(&coords, 2)
}

/// Garble the maps `a_j·x + b_j mod p_i` for every prime `p_i` of `params`.
///
/// `a_residues[i][j]` and `b_residues[i][j]` are the coefficients of map `j < S`
/// reduced mod `p_i`. Returns the program and the encoding information.
pub fn garble<R: Rng>(
    rng: &mut R,
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> (Program, EncodingInfo) {
    let n = params.n as usize;
    assert_eq!(a_residues.len(), params.num_primes);
    assert_eq!(b_residues.len(), params.num_primes);
    let s_dim = a_residues[0].len();
    for i in 0..params.num_primes {
        assert_eq!(a_residues[i].len(), s_dim);
        assert_eq!(b_residues[i].len(), s_dim);
    }
    let ell = params.ell;
    let chunk_size = params.chunk_size as usize;
    let num_chunks = params.num_chunks;
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);
    let first_width = sub_widths[0];

    let delta: u128 = rng.random::<u128>() | 1;
    let d2 = label::delta_r(delta, 2);
    let layout = NonceLayout::new(params, s_dim, &sub_widths);

    let bit_masks: Vec<Label> = (0..n).map(|_| sample_bit_mask(rng)).collect();
    let mut program_bits = 0usize;
    let mut mask_bits = 0usize;

    // Stage 1: chunk conversion (garbler side).
    let zero_bit_mask = Label::zero(2);
    let mut chunk_word_masks: Vec<Label> = Vec::with_capacity(num_chunks);
    let mut chunks = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let mut masks: Vec<Label> = bit_masks[start..end].to_vec();
        while masks.len() < chunk_size {
            masks.push(zero_bit_mask.clone());
        }
        let bulk = layout.bulk_chunk_base + c as u64 * layout.chunk_bulk_ids;
        let solo = layout.solo_chunk_base + c as u64 * layout.chunk_solo_ids;
        let g = chunk_batch_garble(&masks, ell, delta, bulk, solo);
        program_bits += g.cost.program_bits;
        chunk_word_masks.push(g.word_mask);
        chunks.push(ChunkCiphertexts {
            scale: g.scale,
            pin: g.pin,
        });
    }

    // Stages 2..4 per prime (garbler side).
    let mut primes = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        let coeffs: Vec<u64> = (0..num_chunks)
            .map(|c| pow2_mod((c * chunk_size) as u32, p_i))
            .collect();
        let bulk = layout.bulk_extract_base + i as u64 * layout.ex_bulk_ids;
        let solo = layout.solo_extract_base + i as u64 * layout.ex_solo_ids;
        let g = extract_batch_garble(&chunk_word_masks, &coeffs, &sub_widths, delta, bulk, solo);
        program_bits += g.cost.program_bits;

        let fold_nonce_base = layout.prime_nonce_bases[i] + layout.num_batches as u64 * p_i;
        let fold_g = fold_batch_garble(
            p_i,
            &g.first_bin_hot_masks,
            &g.fold_bit_masks,
            first_width,
            fold_nonce_base,
        );
        program_bits += fold_g.cost.program_bits;

        let weights: Vec<u64> = (0..p_i).collect();
        let mut body = Vec::new();
        let mut start = 0usize;
        while start < s_dim {
            let end = (start + RESIDUE_BATCH_SIZE).min(s_dim);
            let a_batch: Vec<u64> = a_residues[i][start..end].iter().map(|&a| a % p_i).collect();
            let b_batch: Vec<u64> = b_residues[i][start..end].iter().map(|&b| b % p_i).collect();
            let batch_idx = start / RESIDUE_BATCH_SIZE;
            let nonce_base = layout.prime_nonce_bases[i] as usize + batch_idx * p_i as usize;
            let g_out = body_batch_garble(
                p_i,
                &fold_g.h_p_masks,
                &a_batch,
                &b_batch,
                &weights,
                nonce_base,
            );
            program_bits += g_out.cost.program_bits;
            mask_bits += g_out.result_masks.len() * lg_modulus(p_i);
            body.push(BodyBatch {
                join_diffs: g_out.join_diffs,
                result_masks: g_out.result_masks,
            });
            start = end;
        }
        primes.push(PrimeCiphertexts {
            extract: g.diffs,
            fold: fold_g.join_diffs,
            body,
        });
    }

    (
        Program {
            params: params.clone(),
            s_dim,
            chunks,
            primes,
            program_bits,
            mask_bits,
        },
        EncodingInfo { bit_masks, d2 },
    )
}

/// Same as [`garble`], with the garbler's coins expanded from a 256-bit seed.
/// Callers whose randomness source is not a [`rand::Rng`] of this crate's
/// `rand` version draw the seed from their own generator.
pub fn garble_from_seed(
    seed: [u8; 32],
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> (Program, EncodingInfo) {
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    garble(&mut rng, params, a_residues, b_residues)
}

/// The projective encoding: for each bit, the mask or the mask plus `Δ₂`.
pub fn encode(ek: &EncodingInfo, x_bits: &[u64]) -> InputLabels {
    assert_eq!(x_bits.len(), ek.bit_masks.len());
    let labels = ek
        .bit_masks
        .iter()
        .zip(x_bits)
        .map(|(m, &b)| {
            assert!(b < 2, "x_bits entries must be 0 or 1");
            if b == 1 {
                label::add(m, &ek.d2)
            } else {
                m.clone()
            }
        })
        .collect();
    InputLabels { labels }
}

/// Evaluate: returns `out[i][j] = a_j·x + b_j mod p_i` for prime `i` and map
/// `j`. `x_bits` are the evaluator's bits in the clear (LSB first).
pub fn eval(program: &Program, labels: &InputLabels, x_bits: &[u64]) -> Vec<Vec<u64>> {
    let params = &program.params;
    let n = params.n as usize;
    assert_eq!(x_bits.len(), n);
    assert_eq!(labels.labels.len(), n);
    for &b in x_bits {
        assert!(b < 2, "x_bits entries must be 0 or 1");
    }
    let ell = params.ell;
    let chunk_size = params.chunk_size as usize;
    let work_mod = 1u64 << ell;
    let num_chunks = params.num_chunks;
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);
    let first_width = sub_widths[0];
    let layout = NonceLayout::new(params, program.s_dim, &sub_widths);

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

    // Stage 1 (evaluator side).
    let zero_bit_label = Label::zero(2);
    let mut chunk_word_labels: Vec<Label> = Vec::with_capacity(num_chunks);
    for (c, ct) in program.chunks.iter().enumerate() {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let mut ls: Vec<Label> = labels.labels[start..end].to_vec();
        while ls.len() < chunk_size {
            ls.push(zero_bit_label.clone());
        }
        let bulk = layout.bulk_chunk_base + c as u64 * layout.chunk_bulk_ids;
        let solo = layout.solo_chunk_base + c as u64 * layout.chunk_solo_ids;
        let w = chunk_batch_eval(&ls, chunk_values[c], ell, &ct.scale, &ct.pin, bulk, solo);
        chunk_word_labels.push(w);
    }

    // Stages 2..4 per prime (evaluator side).
    let mut outputs: Vec<Vec<u64>> = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        let pc = &program.primes[i];
        let coeffs: Vec<u64> = (0..num_chunks)
            .map(|c| pow2_mod((c * chunk_size) as u32, p_i))
            .collect();
        let r_value: u64 = chunk_values.iter().zip(&coeffs).map(|(&v, &c)| c * v).sum();
        assert!(r_value < work_mod, "r_i overflows 2^ell");
        let hot_i = (r_value % p_i) as usize;

        let bulk = layout.bulk_extract_base + i as u64 * layout.ex_bulk_ids;
        let solo = layout.solo_extract_base + i as u64 * layout.ex_solo_ids;
        let ex = extract_batch_eval(
            &chunk_word_labels,
            &coeffs,
            r_value,
            &sub_widths,
            &pc.extract,
            bulk,
            solo,
        );
        let fold_nonce_base = layout.prime_nonce_bases[i] + layout.num_batches as u64 * p_i;
        let h_p_labels = fold_batch_eval(
            p_i,
            r_value,
            &ex.first_bin_hot_labels,
            &ex.fold_bit_labels,
            &pc.fold,
            first_width,
            fold_nonce_base,
        );

        let weights: Vec<u64> = (0..p_i).collect();
        let mut prime_outputs: Vec<u64> = Vec::with_capacity(program.s_dim);
        for (batch_idx, batch) in pc.body.iter().enumerate() {
            let nonce_base = layout.prime_nonce_bases[i] as usize + batch_idx * p_i as usize;
            // `body_batch_eval` reads `b_batch` for its length only.
            let b_len = vec![0u64; batch.join_diffs.len()];
            let result_labels = body_batch_eval(
                p_i,
                hot_i,
                &h_p_labels,
                &batch.join_diffs,
                &b_len,
                &weights,
                nonce_base,
            );
            for (l, m) in result_labels.iter().zip(batch.result_masks.iter()) {
                let value = if l >= m { l - m } else { l + p_i - m };
                prime_outputs.push(value);
            }
        }
        outputs.push(prime_outputs);
    }
    outputs
}

/// Transpose the output of [`eval`] to `out[j][i]`, the residues of map `j`
/// across the primes, which is the shape CRT reconstruction consumes.
pub fn by_component(outputs: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let s_dim = outputs.first().map_or(0, Vec::len);
    (0..s_dim)
        .map(|j| outputs.iter().map(|row| row[j]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crt::bigint::FIRST_80_PRIMES;

    /// The split API agrees with the direct mod-p oracle, including at the
    /// production shape (n = 256, the 80-prime set).
    #[test]
    fn test_pgs_split_matches_oracle() {
        let mut rng = rand::rng();
        for (n, s_dim) in [(8u32, 3usize), (33, 2), (64, 5), (256, 7)] {
            let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
            let a_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let b_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let a_res: Vec<Vec<u64>> = params
                .primes
                .iter()
                .map(|&p| a_vals.iter().map(|&a| a % p).collect())
                .collect();
            let b_res: Vec<Vec<u64>> = params
                .primes
                .iter()
                .map(|&p| b_vals.iter().map(|&b| b % p).collect())
                .collect();
            // x: a random n-bit value; for n > 64 only the low 64 bits are nonzero,
            // which keeps the oracle in u128 arithmetic.
            let x: u64 = if n >= 64 {
                rng.random::<u64>()
            } else {
                rng.random_range(0..1u64 << n)
            };
            let x_bits: Vec<u64> = (0..n as usize)
                .map(|j| if j < 64 { (x >> j) & 1 } else { 0 })
                .collect();
            let (program, ek) = garble(&mut rng, &params, &a_res, &b_res);
            let labels = encode(&ek, &x_bits);
            let out = eval(&program, &labels, &x_bits);
            for (i, &p_i) in params.primes.iter().enumerate() {
                let xm = (x % p_i) as u128;
                for j in 0..s_dim {
                    let expected =
                        ((a_vals[j] % p_i) as u128 * xm + (b_vals[j] % p_i) as u128) % p_i as u128;
                    assert_eq!(out[i][j] as u128, expected, "prime {p_i}, map {j}, n {n}");
                }
            }
            let t = by_component(&out);
            assert_eq!(t.len(), s_dim);
            assert_eq!(t[0].len(), params.num_primes);
            assert!(program.program_bits() > 0 && program.mask_bits() > 0);
        }
    }

    /// The split API and [`crate::affine::build_s_aff`] are the same protocol:
    /// from the same generator state they emit identical residues on a random
    /// full-width input at the production shape (n = 256, 80 primes).
    #[test]
    fn test_pgs_split_matches_build_s_aff() {
        use crate::affine::build_s_aff;
        use rand::SeedableRng;
        let mut seed_rng = rand::rng();
        for n in [16u32, 64, 256] {
            let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
            let s_dim = 5;
            let a_res: Vec<Vec<u64>> = params
                .primes
                .iter()
                .map(|&p| (0..s_dim).map(|_| seed_rng.random_range(0..p)).collect())
                .collect();
            let b_res: Vec<Vec<u64>> = params
                .primes
                .iter()
                .map(|&p| (0..s_dim).map(|_| seed_rng.random_range(0..p)).collect())
                .collect();
            let x_bits: Vec<u64> = (0..n as usize)
                .map(|_| seed_rng.random_range(0..2u64))
                .collect();
            let seed: [u8; 32] = seed_rng.random();
            let mut r1 = rand::rngs::StdRng::from_seed(seed);
            let mut r2 = rand::rngs::StdRng::from_seed(seed);
            let (reference, _) = build_s_aff(&mut r1, &x_bits, &params, &a_res, &b_res);
            let (program, ek) = garble(&mut r2, &params, &a_res, &b_res);
            let labels = encode(&ek, &x_bits);
            assert_eq!(labels.len(), n as usize);
            let out = eval(&program, &labels, &x_bits);
            assert_eq!(out, reference, "n = {n}");
        }
    }
}
