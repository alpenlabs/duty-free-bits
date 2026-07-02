//! Streaming switch system for evaluating affine maps over a primorial ring.
//!
//! Given an n-bit input x and coefficients (a, b) reduced mod each CRT prime,
//! computes a·x + b in Z_M (where M = Π p_i) via three phases:
//!
//!   1. Chunk conversion: partition n bits into ⌈n/lg n⌉ chunks, convert each to
//!      a word in Z_{2^ℓ} via `bin_to_word`.
//!   2. Free accumulation + fold: for each prime p_i, compute r_i ≡ x (mod p_i),
//!      sub-chunk-extract, and fold to a length-p_i binary one-hot `h_p` of
//!      x mod p_i (the computational GC).
//!   3. Residue evaluation: the information-theoretic GC ([`crate::it_gc`])
//!      delivers a · (x mod p_i) + b from `h_p`.
//!
//! Two drivers run these phases:
//!
//! * [`build_s_aff_kernels`] — the production path: four straight-line
//!   kernels (chunk conversion, extract, fold, body) over bare labels, with
//!   no `System`, no `Pipeline`, and no worklist anywhere.
//! * [`build_s_aff_streaming`] — the `System`-phase path via a [`Pipeline`],
//!   kept as the reference implementation the kernels are differentially
//!   tested against.

use crate::comp_gc::convert::{bin_to_word, compute_sub_widths, sub_chunk_extract};
use crate::comp_gc::extract::{
    chunk_batch_garble, chunk_batch_eval, chunk_kernel_nonces, extract_batch_eval,
    extract_batch_garble, extract_kernel_nonces,
};
use crate::comp_gc::fold::{fold_batch_eval, fold_batch_garble};
use crate::crt::{CrtParams, pow2_mod};
use crate::crypto::nonce;
use crate::it_gc::{body_batch_eval, body_batch_garble};
use crate::label::{self, Label};
use crate::pipeline::{CarryId, Pipeline, sample_cf_mask};
use crate::types::*;

/// Maximum sub-chunk width for the sub-chunk extraction optimization.
/// 2^8 = 256 OHE entries per sub-chunk.
const MAX_SUB_CHUNK_WIDTH: u32 = 8;

/// S-batch size for the streaming residue body.
///
/// Splitting the S components into batches keeps each phase's working set small.
/// 128 fills a full λ-bit CCRH block (so packing isn't wasted) while keeping
/// per-phase peak wires low.
const RESIDUE_BATCH_SIZE: usize = 128;

/// First bulk-domain CCRH id available to the kernels (body + fold; the
/// chunk/extract tree hashes allocate above their windows).
///
/// `[0, 2^32)` is reserved for in-System NCF switch-group indices
/// ([`crate::system::System::register_ncf_switch_group`]), which share the
/// bulk domain — keeping the spaces disjoint preserves Definition-4 nonce
/// freshness even if a future phase registers switch groups.
const KERNEL_NONCE_FLOOR: u64 = 1 << 32;

/// Garble + evaluate the affine maps as a sequence of independent phases via a
/// [`Pipeline`]:
///
/// * **One phase per chunk** for `bin_to_word`. Carry: the chunk word `w_c`.
/// * **Per prime, an extract phase** that builds `r_i` and runs
///   `sub_chunk_extract`. Carry: the first sub-chunk's binary OHE plus the
///   remaining bit wires.
/// * **Per prime, a fold kernel** ([`crate::comp_gc::fold`]) consuming those
///   carries' masks/labels directly and producing the length-`p_i` OHE `h_p`
///   without a System.
/// * **Per prime, body sub-phases** (one per `RESIDUE_BATCH_SIZE`-batch of S),
///   each consuming `h_p` + a batch of `(a, b)` coefficients and producing
///   that batch's decoded outputs (via [`crate::it_gc`]).
///
/// After each sub-phase the System (gates, masks, labels, program) is dropped —
/// only the small carry-forward set survives, satisfying
/// `label = mask + value · Δ_R(modulus)` across boundaries.
///
/// `x_bits` are the cleartext input bits, known to the evaluator in the
/// privacy-preserving switch-private/data-public setting. They derive
/// `hot_i = x mod p_i` directly for the body kernel; switches reveal nothing.
///
/// **Statistical smudging (paper Thm. 5.2):** this function evaluates the
/// affine maps exactly as given. When the evaluator will CRT-reconstruct
/// `a·x + b` over Z_M, the garbler must pre-smudge each `b` as
/// `b' = b + μ·p` with `μ` uniform over a 2^ρ-sized domain *before* deriving
/// `b_residues` — otherwise the reconstructed integer leaks more about
/// `(a, b)` than `a·x + b mod p` does. Smudging is the caller's
/// responsibility; it is parameter preparation, not part of the switch
/// system.
///
/// `input_bit_ids` are carry ids for the n input bits (seed them with
/// [`Pipeline::seed_input_cf_value`]).
///
/// [`CarryItem`]: crate::pipeline::CarryItem
pub fn build_s_aff_streaming(
    pipeline: &mut Pipeline,
    input_bit_ids: &[CarryId],
    x_bits: &[u64],
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    let n = input_bit_ids.len();
    assert_eq!(n, params.n as usize);
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

    // ---- Phase set 1: chunk conversion. One phase per chunk c. ----
    let mut chunk_word_ids: Vec<CarryId> = Vec::with_capacity(params.num_chunks);
    for c in 0..params.num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let chunk_input_ids = &input_bit_ids[start..end];

        // All full chunks share one System shape; a short last chunk (padded
        // with a constant wire) is its own shape.
        let outs = pipeline.run_phase_keyed(
            format!("chunk[{c}]"),
            &format!("chunk/{}", chunk_input_ids.len()),
            chunk_input_ids,
            move |sys, chunk_wires| {
                let mut bits: Vec<Wire> = chunk_wires.to_vec();
                if bits.len() < chunk_size {
                    let zero_bit = sys.constant(0, 2);
                    while bits.len() < chunk_size {
                        bits.push(zero_bit);
                    }
                }
                vec![bin_to_word(sys, &bits, ell)]
            },
        );
        chunk_word_ids.push(outs[0]);
    }

    // ---- Phase set 2: per prime, a System extract phase + two kernels. ----
    //
    // The extract phase builds `r_i` and its sub-chunk decomposition on the
    // System path; the fold to the length-`p_i` OHE and the residue body then
    // run on straight-line kernels ([`crate::comp_gc::fold`], [`crate::it_gc`]).
    // The evaluator knows `x_bits`, so it derives `hot_i = x mod p_i` and every
    // switch control itself — no switch reveals anything (extract, fold or body).
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);
    let first_width = sub_widths[0];
    let fold_bits: u32 = sub_widths[1..].iter().sum();
    let mut all_outputs: Vec<Vec<u64>> = Vec::with_capacity(params.num_primes);

    // Chunk values of x (the evaluator's view): w_c = x bits [c·cs, (c+1)·cs).
    let chunk_values: Vec<u64> = (0..params.num_chunks)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            x_bits[start..end]
                .iter()
                .enumerate()
                .fold(0u64, |acc, (j, &b)| acc | (b << j))
        })
        .collect();

    // CCRH nonce windows, one per prime: `num_batches·p` body ids (a body
    // switch's pad is `H(h_p[i], nonce)`) followed by `fold_bits·p` fold ids.
    let num_batches = s_dim.div_ceil(RESIDUE_BATCH_SIZE);
    let prime_window_sizes: Vec<u64> = params
        .primes
        .iter()
        .map(|&p| (num_batches as u64 + fold_bits as u64) * p)
        .collect();
    let prime_nonce_bases = nonce::windows(KERNEL_NONCE_FLOOR, &prime_window_sizes);
    // The bulk CCRH domain reserves bit 63; every kernel id must stay below it
    // (hash_z2/hash_bulk only debug-check this).
    if let (Some(&base), Some(&size)) = (prime_nonce_bases.last(), prime_window_sizes.last()) {
        assert!(
            base + size < (1u64 << 63),
            "kernel CCRH nonce space exhausted"
        );
    }

    for (i, &p_i) in params.primes.iter().enumerate() {
        // hot_i = x mod p_i, derived once from x_bits and reused per body batch.
        let hot_i = {
            let mut acc = 0u64;
            let mut weight = 1u64;
            for &bit in x_bits {
                if bit == 1 {
                    acc = (acc + weight) % p_i;
                }
                weight = (weight << 1) % p_i;
            }
            acc as usize
        };
        debug_assert!(hot_i < p_i as usize);

        // The evaluator's cleartext r_i = Σ_c coeff_c·w_c: by construction
        // 2^ell exceeds the worst-case sum (see CrtParams), so no reduction.
        let r_i_value: u64 = chunk_values
            .iter()
            .enumerate()
            .map(|(c, &w)| pow2_mod((c * chunk_size) as u32, p_i) * w)
            .sum();
        assert!(
            r_i_value < work_mod,
            "r_i overflows 2^ell — CrtParams.ell undersized for these primes/n"
        );
        debug_assert_eq!(r_i_value % p_i, hot_i as u64);

        // -- Extract phase: r_i + sub_chunk_extract. Carries out the first
        // sub-chunk's binary OHE and the remaining fold bits.
        // For odd primes every r_i coefficient is nonzero, so the extract
        // System is structurally identical across them (only Mul scalar
        // values differ) and one recorded garble schedule serves all; p = 2
        // skips zero-coefficient gates and garbles unkeyed.
        let sub_widths_local = sub_widths.clone();
        let build_extract = move |sys: &mut crate::system::System, chunk_wires: &[Wire]| {
            let mut r_i = sys.constant(0, work_mod);
            for (c, &w_c) in chunk_wires.iter().enumerate() {
                let coeff = pow2_mod((c * chunk_size) as u32, p_i);
                if coeff > 0 {
                    let term = sys.mul(coeff, w_c);
                    r_i = sys.add(r_i, term);
                }
            }
            let ex = sub_chunk_extract(sys, r_i, &sub_widths_local);
            let mut outs = ex.first_bin_hot;
            outs.extend(ex.bits[1..].iter().flatten().copied());
            outs
        };
        let extract_ids = if p_i == 2 {
            pipeline.run_phase(
                format!("prime[{i}]/extract"),
                &chunk_word_ids,
                build_extract,
            )
        } else {
            pipeline.run_phase_keyed(
                format!("prime[{i}]/extract"),
                "extract/odd",
                &chunk_word_ids,
                build_extract,
            )
        };
        let ohe_len = 1usize << first_width;
        let (fbh_ids, bit_ids) = extract_ids.split_at(ohe_len);

        // -- Fold kernel: h_p = OHE of r_i mod p_i, as masks/labels directly.
        let fold_nonce_base = prime_nonce_bases[i] + num_batches as u64 * p_i;
        let fbh_masks: Vec<_> = fbh_ids
            .iter()
            .map(|&id| pipeline.carry(id).mask.clone())
            .collect();
        let bit_masks: Vec<_> = bit_ids
            .iter()
            .map(|&id| pipeline.carry(id).mask.clone())
            .collect();
        let fold_gb0 = crate::crypto::hash_blocks();
        let t_fold_garble = std::time::Instant::now();
        let fold_g = fold_batch_garble(p_i, &fbh_masks, &bit_masks, first_width, fold_nonce_base);
        let fold_garble_secs = t_fold_garble.elapsed().as_secs_f64();
        pipeline.garble_hash_blocks += crate::crypto::hash_blocks() - fold_gb0;

        let fbh_labels: Vec<_> = fbh_ids
            .iter()
            .map(|&id| pipeline.carry(id).label.clone())
            .collect();
        let bit_labels: Vec<_> = bit_ids
            .iter()
            .map(|&id| pipeline.carry(id).label.clone())
            .collect();
        let fold_eb0 = crate::crypto::hash_blocks();
        let t_fold_eval = std::time::Instant::now();
        let h_p_labels = fold_batch_eval(
            p_i,
            r_i_value,
            &fbh_labels,
            &bit_labels,
            &fold_g.join_diffs,
            first_width,
            fold_nonce_base,
        );
        let fold_eval_secs = t_fold_eval.elapsed().as_secs_f64();
        pipeline.eval_hash_blocks += crate::crypto::hash_blocks() - fold_eb0;
        pipeline.record_fold_batch(fold_garble_secs, fold_eval_secs, fold_g.cost);
        let h_p_masks = fold_g.h_p_masks;

        for &id in extract_ids.iter() {
            pipeline.drop_carry(id);
        }

        // -- Body batches: each consumes h_p + a batch of (a, b) → batch outputs.
        let identity_table: Vec<u64> = (0..p_i).collect();
        let mut prime_outputs: Vec<u64> = Vec::with_capacity(s_dim);

        let mut start = 0usize;
        while start < s_dim {
            let end = (start + RESIDUE_BATCH_SIZE).min(s_dim);
            let a_batch: Vec<u64> = a_residues[i][start..end].iter().map(|&a| a % p_i).collect();
            let b_batch: Vec<u64> = b_residues[i][start..end].iter().map(|&b| b % p_i).collect();
            // This batch's fresh nonce block, drawn from the prime's window.
            let batch_idx = start / RESIDUE_BATCH_SIZE;
            let nonce_base = prime_nonce_bases[i] as usize + batch_idx * p_i as usize;

            // Garbler kernel: emits join diffs, output masks, and the batch cost.
            let body_gb0 = crate::crypto::hash_blocks();
            let t_garble = std::time::Instant::now();
            let g_out = body_batch_garble(
                p_i,
                &h_p_masks,
                &a_batch,
                &b_batch,
                &identity_table,
                nonce_base,
            );
            let garble_secs = t_garble.elapsed().as_secs_f64();
            pipeline.garble_hash_blocks += crate::crypto::hash_blocks() - body_gb0;
            // Evaluator kernel: consumes the garbler's join diffs, using the same
            // nonce base. `hot_i` comes from cleartext `x_bits`; no reveal.
            let body_eb0 = crate::crypto::hash_blocks();
            let t_eval = std::time::Instant::now();
            let result_labels = body_batch_eval(
                p_i,
                hot_i,
                &h_p_labels,
                &g_out.join_diffs,
                &b_batch,
                &identity_table,
                nonce_base,
            );
            let eval_secs = t_eval.elapsed().as_secs_f64();
            pipeline.eval_hash_blocks += crate::crypto::hash_blocks() - body_eb0;

            // Decode: value_j = (label_j − mask_j) mod p_i.
            for (label, mask) in result_labels.iter().zip(g_out.result_masks.iter()) {
                let value = if label >= mask {
                    label - mask
                } else {
                    label + p_i - mask
                };
                prime_outputs.push(value);
            }

            pipeline.record_kernel_batch(garble_secs, eval_secs, g_out.cost);
            start = end;
        }

        all_outputs.push(prime_outputs);
    }

    for id in chunk_word_ids {
        pipeline.drop_carry(id);
    }

    all_outputs
}

// ==================== the kernel-only production path ====================

/// Per-stage telemetry of one [`build_s_aff_kernels`] run. Times are
/// wall-clock seconds; `*_hash_blocks` are CCRH AES blocks (nonzero only
/// under the `count-hashes` feature); the ledger fields are the same units
/// `System::cost` charges the equivalent circuits.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelStats {
    /// Chunk-kernel garble wall time.
    pub chunk_garble_secs: f64,
    /// Chunk-kernel eval wall time.
    pub chunk_eval_secs: f64,
    /// Extract-kernel garble wall time.
    pub extract_garble_secs: f64,
    /// Extract-kernel eval wall time.
    pub extract_eval_secs: f64,
    /// Fold-kernel garble wall time.
    pub fold_garble_secs: f64,
    /// Fold-kernel eval wall time.
    pub fold_eval_secs: f64,
    /// Body-kernel garble wall time.
    pub body_garble_secs: f64,
    /// Body-kernel eval wall time.
    pub body_eval_secs: f64,
    /// Garbler-emitted material in bits.
    pub program_bits: usize,
    /// CF join width, in `lg|G|` units (each pays λ bits).
    pub join_complexity_cf: usize,
    /// NCF join width in bits.
    pub join_complexity_ncf: usize,
    /// Ledger CCRH blocks from CF switches.
    pub hash_count_cf: usize,
    /// Ledger CCRH blocks from NCF switches (bulk-pack rebated).
    pub hash_count_ncf: usize,
    /// Measured garbler CCRH blocks (`count-hashes` feature; else 0).
    pub garble_hash_blocks: u64,
    /// Measured evaluator CCRH blocks (`count-hashes` feature; else 0).
    pub eval_hash_blocks: u64,
}

impl KernelStats {
    /// Total garble/eval wall time.
    pub fn garble_secs(&self) -> f64 {
        self.chunk_garble_secs + self.extract_garble_secs + self.fold_garble_secs
            + self.body_garble_secs
    }
    /// See [`garble_secs`](Self::garble_secs).
    pub fn eval_secs(&self) -> f64 {
        self.chunk_eval_secs + self.extract_eval_secs + self.fold_eval_secs + self.body_eval_secs
    }
}

/// Garble + evaluate the affine maps on the kernel-only path: chunk
/// conversion → per-prime extract → fold → body, all straight-line loops over
/// bare labels ([`crate::comp_gc::extract`], [`crate::comp_gc::fold`],
/// [`crate::it_gc`]). No `System`, no `Pipeline`, no worklist: the evaluator
/// knows `x_bits` (switch-private / data-public), so every derivation order is
/// closed-form.
///
/// Semantics, inputs, and the smudging caveat are those of
/// [`build_s_aff_streaming`]; the two are differentially tested to produce
/// identical decoded outputs. One parameter restriction: the extract kernels
/// require every sub-chunk width in `2..=31`, so shapes with a width-1
/// trailing sub-chunk (`ell ≡ 1 mod 8`, e.g. n = 26..=32 with the 80-prime
/// set) are rejected loudly — use the reference path for those.
pub fn build_s_aff_kernels<R: rand::Rng>(
    rng: &mut R,
    x_bits: &[u64],
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> (Vec<Vec<u64>>, KernelStats) {
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
    let fold_bits: u32 = sub_widths[1..].iter().sum();

    let delta: u128 = rng.random::<u128>() | 1;
    let d2 = Label::Cf(label::delta_r(delta, 2));
    let mut stats = KernelStats::default();

    // ---- Nonce windows (paper App. A, Def. 4) ----
    // Solo domain (width-l casts): chunk kernels first, then extracts.
    let (chunk_bulk_ids, chunk_solo_ids) = chunk_kernel_nonces(chunk_size as u32);
    let (ex_bulk_ids, ex_solo_ids) = extract_kernel_nonces(&sub_widths);
    let solo_chunk_base = 0u64;
    let solo_extract_base = solo_chunk_base + num_chunks as u64 * chunk_solo_ids;
    // Bulk domain: fold + body windows exactly as the streaming path lays
    // them out, then the tree windows above them.
    let num_batches = s_dim.div_ceil(RESIDUE_BATCH_SIZE);
    let prime_window_sizes: Vec<u64> = params
        .primes
        .iter()
        .map(|&p| (num_batches as u64 + fold_bits as u64) * p)
        .collect();
    let prime_nonce_bases = nonce::windows(KERNEL_NONCE_FLOOR, &prime_window_sizes);
    let bulk_tree_base = prime_nonce_bases
        .last()
        .zip(prime_window_sizes.last())
        .map(|(&b, &s)| b + s)
        .unwrap_or(KERNEL_NONCE_FLOOR);
    let bulk_chunk_base = bulk_tree_base;
    let bulk_extract_base = bulk_chunk_base + num_chunks as u64 * chunk_bulk_ids;
    assert!(
        bulk_extract_base + params.num_primes as u64 * ex_bulk_ids < (1u64 << 63),
        "kernel CCRH nonce space exhausted"
    );
    assert!(
        solo_extract_base + params.num_primes as u64 * ex_solo_ids < (1u64 << 63),
        "kernel solo-domain nonce space exhausted"
    );

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

    // ---- Stage 1: chunk conversion kernels. ----
    let zero_bit_mask = Label::zero_cf(2); // constant-0 padding for a short last chunk
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

        let gb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let g = chunk_batch_garble(&masks, ell, delta, bulk, solo);
        stats.chunk_garble_secs += t.elapsed().as_secs_f64();
        stats.garble_hash_blocks += crate::crypto::hash_blocks() - gb0;

        let eb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let w = chunk_batch_eval(&labels, chunk_values[c], ell, &g.scale, &g.pin, bulk, solo);
        stats.chunk_eval_secs += t.elapsed().as_secs_f64();
        stats.eval_hash_blocks += crate::crypto::hash_blocks() - eb0;

        stats.program_bits += g.cost.program_bits;
        stats.join_complexity_cf += g.cost.join_complexity_cf;
        stats.hash_count_cf += g.cost.hash_count_cf;
        chunk_word_masks.push(g.word_mask);
        chunk_word_labels.push(w);
    }

    // ---- Stage 2..4 per prime: extract kernel → fold kernel → body. ----
    let mut all_outputs: Vec<Vec<u64>> = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        let coeffs: Vec<u64> = (0..num_chunks)
            .map(|c| pow2_mod((c * chunk_size) as u32, p_i))
            .collect();
        let r_value: u64 = chunk_values
            .iter()
            .zip(&coeffs)
            .map(|(&v, &c)| c * v)
            .sum();
        assert!(
            r_value < work_mod,
            "r_i overflows 2^ell — CrtParams.ell undersized for these primes/n"
        );
        let hot_i = (r_value % p_i) as usize;

        let bulk = bulk_extract_base + i as u64 * ex_bulk_ids;
        let solo = solo_extract_base + i as u64 * ex_solo_ids;

        let gb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let g = extract_batch_garble(&chunk_word_masks, &coeffs, &sub_widths, delta, bulk, solo);
        stats.extract_garble_secs += t.elapsed().as_secs_f64();
        stats.garble_hash_blocks += crate::crypto::hash_blocks() - gb0;

        let eb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let (fbh_labels, fold_bit_labels) = extract_batch_eval(
            &chunk_word_labels,
            &coeffs,
            r_value,
            &sub_widths,
            &g.diffs,
            bulk,
            solo,
        );
        stats.extract_eval_secs += t.elapsed().as_secs_f64();
        stats.eval_hash_blocks += crate::crypto::hash_blocks() - eb0;

        stats.program_bits += g.cost.program_bits;
        stats.join_complexity_cf += g.cost.join_complexity_cf;
        stats.hash_count_cf += g.cost.hash_count_cf;

        // ---- Fold kernel (unchanged from the streaming path). ----
        let fold_nonce_base = prime_nonce_bases[i] + num_batches as u64 * p_i;
        let gb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let fold_g = fold_batch_garble(
            p_i,
            &g.first_bin_hot_masks,
            &g.fold_bit_masks,
            first_width,
            fold_nonce_base,
        );
        stats.fold_garble_secs += t.elapsed().as_secs_f64();
        stats.garble_hash_blocks += crate::crypto::hash_blocks() - gb0;

        let eb0 = crate::crypto::hash_blocks();
        let t = std::time::Instant::now();
        let h_p_labels = fold_batch_eval(
            p_i,
            r_value,
            &fbh_labels,
            &fold_bit_labels,
            &fold_g.join_diffs,
            first_width,
            fold_nonce_base,
        );
        stats.fold_eval_secs += t.elapsed().as_secs_f64();
        stats.eval_hash_blocks += crate::crypto::hash_blocks() - eb0;
        stats.program_bits += fold_g.cost.program_bits;
        stats.join_complexity_cf += fold_g.cost.join_complexity_cf;
        stats.hash_count_cf += fold_g.cost.hash_count_cf;
        let h_p_masks = fold_g.h_p_masks;

        // ---- Body batches (unchanged). ----
        let identity_table: Vec<u64> = (0..p_i).collect();
        let mut prime_outputs: Vec<u64> = Vec::with_capacity(s_dim);
        let mut start = 0usize;
        while start < s_dim {
            let end = (start + RESIDUE_BATCH_SIZE).min(s_dim);
            let a_batch: Vec<u64> = a_residues[i][start..end].iter().map(|&a| a % p_i).collect();
            let b_batch: Vec<u64> = b_residues[i][start..end].iter().map(|&b| b % p_i).collect();
            let batch_idx = start / RESIDUE_BATCH_SIZE;
            let nonce_base = prime_nonce_bases[i] as usize + batch_idx * p_i as usize;

            let gb0 = crate::crypto::hash_blocks();
            let t = std::time::Instant::now();
            let g_out = body_batch_garble(
                p_i,
                &h_p_masks,
                &a_batch,
                &b_batch,
                &identity_table,
                nonce_base,
            );
            stats.body_garble_secs += t.elapsed().as_secs_f64();
            stats.garble_hash_blocks += crate::crypto::hash_blocks() - gb0;

            let eb0 = crate::crypto::hash_blocks();
            let t = std::time::Instant::now();
            let result_labels = body_batch_eval(
                p_i,
                hot_i,
                &h_p_labels,
                &g_out.join_diffs,
                &b_batch,
                &identity_table,
                nonce_base,
            );
            stats.body_eval_secs += t.elapsed().as_secs_f64();
            stats.eval_hash_blocks += crate::crypto::hash_blocks() - eb0;

            for (l, m) in result_labels.iter().zip(g_out.result_masks.iter()) {
                let value = if l >= m { l - m } else { l + p_i - m };
                prime_outputs.push(value);
            }
            stats.program_bits += g_out.cost.program_bits;
            stats.join_complexity_ncf += g_out.cost.join_complexity_ncf;
            stats.hash_count_ncf += g_out.cost.hash_count_ncf;
            start = end;
        }
        all_outputs.push(prime_outputs);
    }

    (all_outputs, stats)
}
