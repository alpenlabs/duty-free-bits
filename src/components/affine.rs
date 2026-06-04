/// Switch system for evaluating affine maps over a primorial ring.
///
/// Given an n-bit input x and coefficients (a, b) reduced mod each CRT prime,
/// computes a·x + b in Z_M (where M = Π p_i) via three steps:
///
///   1. Chunk conversion: partition n bits into ⌈n/lg n⌉ chunks, convert each
///      to a word in Z_{2^ℓ} via `bin_to_word`.
///   2. Free accumulation: for each prime p_i, compute r_i ≡ x (mod p_i) in
///      Z_{2^ℓ} as a weighted sum of chunk words.
///   3. Residue evaluation: for each prime, decompose the ℓ-bit residue via
///      sub-chunk extraction into a length-p_i OHE of r_i mod p_i,
///      then evaluate a · (r_i mod p_i) + b via `hot_to_ring`.
use crate::it_gc::{body_batch_eval, body_batch_garble};
use super::convert::{
    bin_to_word, compute_sub_widths, fold_to_mod_ohe, hot_to_ring_bulk, sub_chunk_extract,
};
use super::crt::{CrtParams, pow2_mod};
use crate::crypto::nonce;
use crate::garble::{CarryId, Pipeline};
use crate::system::System;
use crate::types::*;

/// Maximum sub-chunk width for the sub-chunk extraction optimization.
/// 2^8 = 256 OHE entries per sub-chunk.
const MAX_SUB_CHUNK_WIDTH: u32 = 8;

/// Output of the affine switch system: `outputs[i][j]` is the j-th component's
/// result reduced mod the i-th CRT prime.
#[derive(Debug)]
pub struct AffineOutput {
    /// `outputs[i][j]` is a wire in Z_{p_i} for the j-th affine component.
    pub outputs: Vec<Vec<Wire>>,
}

/// Chunk the n-bit input and accumulate CRT residues (steps 1–2).
///
/// Returns one wire per prime in Z_{2^ℓ}, holding a value ≡ x (mod p_i).
pub fn chunk_and_accumulate(
    sys: &mut System,
    input_bits: &[Wire],
    params: &CrtParams,
) -> Vec<Wire> {
    let n = input_bits.len();
    assert_eq!(n, params.n as usize);

    let ell = params.ell;
    let chunk_size = params.chunk_size as usize;
    let work_mod = 1u64 << ell;

    // Step 1: partition into chunks, convert each to a word in Z_{2^ℓ}.
    let mut chunks: Vec<Wire> = Vec::with_capacity(params.num_chunks);
    for c in 0..params.num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n);
        let chunk_bits = &input_bits[start..end];

        if chunk_bits.len() < chunk_size {
            let mut padded = chunk_bits.to_vec();
            let zero_bit = sys.constant(0, 2);
            while padded.len() < chunk_size {
                padded.push(zero_bit);
            }
            chunks.push(bin_to_word(sys, &padded, ell));
        } else {
            chunks.push(bin_to_word(sys, chunk_bits, ell));
        }
    }

    // Step 2: for each prime, r_i = Σ_c (2^{c·chunk_size} mod p_i) · w_c.
    let mut residues = Vec::with_capacity(params.num_primes);
    for &p_i in &params.primes {
        let mut r_i = sys.constant(0, work_mod);
        for (c, &w_c) in chunks.iter().enumerate() {
            let coeff = pow2_mod((c * chunk_size) as u32, p_i);
            if coeff > 0 {
                let term = sys.mul(coeff, w_c);
                r_i = sys.add(r_i, term);
            }
        }
        residues.push(r_i);
    }

    residues
}

/// Build the full affine switch system over the primorial ring.
///
/// Evaluates S affine maps a_j·x + b_j in Z_M for each component j.
/// `a_residues[i][j]` and `b_residues[i][j]` are the j-th component's
/// coefficients reduced mod the i-th CRT prime.
pub fn build_s_aff(
    sys: &mut System,
    input_bits: &[Wire],
    params: &CrtParams,
    a_residues: &[Vec<u64>],
    b_residues: &[Vec<u64>],
) -> AffineOutput {
    assert_eq!(a_residues.len(), params.num_primes);
    assert_eq!(b_residues.len(), params.num_primes);
    let s_dim = a_residues[0].len();
    for i in 0..params.num_primes {
        assert_eq!(a_residues[i].len(), s_dim);
        assert_eq!(b_residues[i].len(), s_dim);
    }

    let ell = params.ell;
    let residue_wires = chunk_and_accumulate(sys, input_bits, params);

    // Step 3: for each prime, evaluate a · (r_i mod p_i) + b via sub-chunk extraction.
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);

    let mut all_outputs = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        // Phase 1 + 2: shared across all S components for this prime.
        let extraction = sub_chunk_extract(sys, residue_wires[i], &sub_widths);
        let h_p = fold_to_mod_ohe(sys, &extraction, p_i);

        let identity_table: Vec<u64> = (0..p_i).collect();
        // Force NCF for the coefficient constants so the pipeline propagates
        // NCF all the way to the output — necessary when p_i = 2 (Z_2 is
        // power-of-two and would otherwise default to CF). The bulk variant
        // packs the S switches sharing each `h_p[k]` into one CCRH call.
        let a_wires: Vec<Wire> = (0..s_dim)
            .map(|j| sys.constant_ncf(a_residues[i][j] % p_i, p_i))
            .collect();
        let b_wires: Vec<Wire> = (0..s_dim)
            .map(|j| sys.constant_ncf(b_residues[i][j] % p_i, p_i))
            .collect();
        let prime_outputs = hot_to_ring_bulk(sys, &h_p, &identity_table, &a_wires, &b_wires);
        all_outputs.push(prime_outputs);
    }

    AffineOutput {
        outputs: all_outputs,
    }
}

/// S-batch size for the streaming residue body.
///
/// Splitting the S components into batches keeps each phase's working set small.
/// 128 fills a full λ-bit CCRH block (so packing isn't wasted) while keeping
/// per-phase peak wires far below the all-at-once path.
const RESIDUE_BATCH_SIZE: usize = 128;

/// Streaming garble + eval of [`build_s_aff`].
///
/// Drives the same algorithm as `build_s_aff` but as a sequence of independent
/// phases via a [`Pipeline`]:
///
/// * **One phase per chunk** for `bin_to_word`. Carry: the chunk word `w_c`.
/// * **Per prime, a header phase** that builds `r_i`, runs `sub_chunk_extract`,
///   and folds to a length-`p_i` OHE `h_p`. Carry: the `p_i` entries of `h_p`.
/// * **Per prime, body sub-phases** (one per `RESIDUE_BATCH_SIZE`-batch of S),
///   each consuming `h_p` + a batch of `(a, b)` coefficients and producing
///   that batch's decoded outputs.
///
/// After each sub-phase the System (gates, masks, labels, program) is dropped —
/// only the small carry-forward `(mask, label)` set survives. The cross-phase
/// invariant `label = mask + value · Δ_R(modulus)` is carried in [`CarryItem`],
/// so outputs are bit-identical to the all-at-once path.
///
/// `x_bits` are the cleartext input bits, known to the evaluator in the
/// privacy-preserving switch-private/data-public setting. They're used to
/// derive `hot_i = x mod p_i` directly, sparing the body kernel from
/// per-switch point-and-permute LSB emission.
///
/// `input_bit_ids` are carry ids for the n input bits (seed them with
/// [`Pipeline::seed_input_cf_value`]).
///
/// [`CarryItem`]: crate::garble::CarryItem
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

    // ---- Phase set 2: per prime, a System header phase + kernel body batches. ----
    //
    // The header builds the OHE `h_p` on the System path (so its masks/labels are
    // pseudorandom under the real CCRH); the body batches then run on the kernel.
    // Because the evaluator knows `x_bits`, it derives `hot_i = x mod p_i`
    // directly and the body never emits a per-switch ctrl LSB.
    let sub_widths = compute_sub_widths(ell, MAX_SUB_CHUNK_WIDTH);
    let mut all_outputs: Vec<Vec<u64>> = Vec::with_capacity(params.num_primes);

    // CCRH nonce windows for the body switches. A body switch's pad is
    // `H(h_p[i], nonce)`; reusing a `(seed, nonce)` pair across batches reuses the
    // one-time pad masking `a_j` — a two-time-pad break leaking differences of the
    // secret coefficients (paper §1.3). We give each prime a disjoint window
    // (`num_batches · p_i` nonces) computed up front, so legality holds by
    // construction and primes are independent (no shared running counter).
    let num_batches = s_dim.div_ceil(RESIDUE_BATCH_SIZE);
    let prime_window_sizes: Vec<u64> = params
        .primes
        .iter()
        .map(|&p| num_batches as u64 * p)
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

        // -- Header: r_i + sub_chunk_extract + fold → h_p (p_i Z_2 CF wires) --
        let sub_widths_local = sub_widths.clone();
        let h_p_ids = pipeline.run_phase(
            format!("prime[{i}]/header"),
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
                let extraction = sub_chunk_extract(sys, r_i, &sub_widths_local);
                fold_to_mod_ohe(sys, &extraction, p_i)
            },
        );

        // Snapshot h_p masks/labels once per prime; the kernel calls per body
        // batch borrow them.
        let h_p_masks: Vec<_> = h_p_ids
            .iter()
            .map(|&id| pipeline.carry(id).mask.clone())
            .collect();
        let h_p_labels: Vec<_> = h_p_ids
            .iter()
            .map(|&id| pipeline.carry(id).label.clone())
            .collect();

        // -- Body batches: each consumes h_p + a batch of (a, b) → batch outputs.
        //    Run on the System-bypass kernel rather than Pipeline::run_phase. --
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
            let g_out =
                body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &identity_table, nonce_base);
            let garble_secs = t_garble.elapsed().as_secs_f64();
            // Evaluator kernel: consumes the garbler's join diffs, using the same
            // nonce base. `hot_i` comes from cleartext `x_bits`, so no ctrl LSB.
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

        for id in h_p_ids {
            pipeline.drop_carry(id);
        }
        all_outputs.push(prime_outputs);
    }

    for id in chunk_word_ids {
        pipeline.drop_carry(id);
    }

    all_outputs
}
