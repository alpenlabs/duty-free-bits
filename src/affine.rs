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
//! [`build_s_aff_streaming`] runs these phase-by-phase via a [`Pipeline`],
//! dropping intermediate state at each boundary.

use crate::comp_gc::convert::{bin_to_word, compute_sub_widths, sub_chunk_extract};
use crate::comp_gc::fold::{fold_batch_eval, fold_batch_garble};
use crate::crt::{CrtParams, pow2_mod};
use crate::crypto::nonce;
use crate::it_gc::{body_batch_eval, body_batch_garble};
use crate::pipeline::{CarryId, Pipeline};
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
        debug_assert!(b < 2);
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

        let outs = pipeline.run_phase(
            format!("chunk[{c}]"),
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
    let prime_nonce_bases = nonce::windows(0, &prime_window_sizes);

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
        debug_assert!(r_i_value < work_mod);
        debug_assert_eq!(r_i_value % p_i, hot_i as u64);

        // -- Extract phase: r_i + sub_chunk_extract. Carries out the first
        // sub-chunk's binary OHE and the remaining fold bits.
        let sub_widths_local = sub_widths.clone();
        let extract_ids = pipeline.run_phase(
            format!("prime[{i}]/extract"),
            &chunk_word_ids,
            move |sys, chunk_wires| {
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
            },
        );
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
        let t_fold_garble = std::time::Instant::now();
        let fold_g = fold_batch_garble(p_i, &fbh_masks, &bit_masks, first_width, fold_nonce_base);
        let fold_garble_secs = t_fold_garble.elapsed().as_secs_f64();

        let fbh_labels: Vec<_> = fbh_ids
            .iter()
            .map(|&id| pipeline.carry(id).label.clone())
            .collect();
        let bit_labels: Vec<_> = bit_ids
            .iter()
            .map(|&id| pipeline.carry(id).label.clone())
            .collect();
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
            // Evaluator kernel: consumes the garbler's join diffs, using the same
            // nonce base. `hot_i` comes from cleartext `x_bits`; no reveal.
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
