/// Switch system for evaluating affine maps over a primorial ring.
///
/// Given an n-bit input x and coefficients (a, b) reduced mod each CRT prime,
/// computes a·x + b in Z_M (where M = Π p_i) via three steps:
///
///   1. Chunk conversion: partition n bits into ⌈n/lg n⌉ chunks, convert each
///      to a word in Z_{2^ℓ} via `bin_to_word`.
///   2. Free accumulation: for each prime p_i, compute r_i ≡ x (mod p_i) in
///      Z_{2^ℓ} as a weighted sum of chunk words.
///   3. Residue evaluation: for each prime, apply `word_to_ring` with the
///      reduction function g(r) = r mod p_i, scaling by a and offsetting by b.
use super::convert::{bin_to_word, word_to_ring};
use super::crt::{CrtParams, pow2_mod};
use crate::system::System;
use crate::types::*;

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

    // Step 3: for each prime, evaluate a · (r_i mod p_i) + b via word_to_ring.
    let mut all_outputs = Vec::with_capacity(params.num_primes);
    for (i, &p_i) in params.primes.iter().enumerate() {
        let table_size = 1usize << ell;
        // TODO: optimize word_to_ring for specifically computing mod.
        let truth_table: Vec<u64> = (0..table_size).map(|r| (r as u64) % p_i).collect();

        let mut prime_outputs = Vec::with_capacity(s_dim);
        for j in 0..s_dim {
            let a_wire = sys.constant(a_residues[i][j] % p_i, p_i);
            let b_wire = sys.constant(b_residues[i][j] % p_i, p_i);
            prime_outputs.push(word_to_ring(
                sys,
                residue_wires[i],
                &truth_table,
                a_wire,
                b_wire,
            ));
        }
        all_outputs.push(prime_outputs);
    }

    AffineOutput {
        outputs: all_outputs,
    }
}

/// Convenience wrapper for a single affine map a·x + b.
pub fn build_s_aff_single(
    sys: &mut System,
    input_bits: &[Wire],
    params: &CrtParams,
    a: u64,
    b: u64,
) -> AffineOutput {
    let a_res: Vec<Vec<u64>> = params.primes.iter().map(|&p| vec![a % p]).collect();
    let b_res: Vec<Vec<u64>> = params.primes.iter().map(|&p| vec![b % p]).collect();
    build_s_aff(sys, input_bits, params, &a_res, &b_res)
}
