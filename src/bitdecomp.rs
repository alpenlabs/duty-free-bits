//! Garbling of `a·x + b` via bit-decomposition, using no one-hot scaling — a
//! standalone, runtime-optimized baseline for benchmarking against the CRT
//! one-hot-scaling path.
//!
//! This module deliberately does **not** use one-hot scaling, the CRT, or any
//! of the engines in [`crate::gc`] / [`crate::gc::body`]. It implements the
//! naive bit-decomposition construction (the \[EL26\]/BABE-style projectivization
//! that the CRT path in [`crate::affine`] replaces) so the two can be compared
//! on the same `a·x+b` workload: the bit-decomposition path pays far more
//! *communication* (`Θ(S·n²)`) but is almost pure CCRH, so its *runtime* is the
//! baseline against which the one-hot-scaling pipeline's extra computation is
//! measured.
//!
//! # Field operations in the workload
//!
//! The garble/eval workload uses only field **add/sub** and powers-of-two
//! **doubling** (the `Σ 2ⁱ` weighting and its Horner recombination). It never
//! multiplies two unknowns. General field multiplication appears only in the
//! final proof verification — out of scope here — so it lives solely in the
//! test oracle and is never on the benchmarked path.
//!
//! # Construction
//!
//! We compute `S` affine maps `aⱼ·x + bⱼ` over `F_p` (BN254 base field), where
//! `x ∈ F_p` is the evaluator's single scalar input, decomposed into
//! `n = N_BITS` bits `x = Σ 2ⁱ xᵢ`. Writing each `bⱼ` as a random weighted
//! additive sharing `bⱼ = Σ 2ⁱ bᵢⱼ` (sample all but the weight-1 share `b₀ⱼ`,
//! then set `b₀ⱼ` to satisfy the sum), the map becomes
//!
//! ```text
//! aⱼ·x + bⱼ = Σ 2ⁱ (xᵢ·aⱼ + bᵢⱼ).
//! ```
//!
//! Because `aⱼ` is known to the garbler and `xᵢ` is a bit, `xᵢ·aⱼ + bᵢⱼ` is a
//! **precomputed selection** — `bᵢⱼ` if `xᵢ=0`, `aⱼ+bᵢⱼ` if `xᵢ=1`. The garbler
//! garbles, per input bit `i`, a two-row table over the whole `S`-vector: row 0
//! masks `{bᵢⱼ}ⱼ` under a counter-mode expansion of `H(Lᵢ⁰)`, row 1 masks
//! `{aⱼ+bᵢⱼ}ⱼ` under `H(Lᵢ¹)` (`Lᵢ¹ = Lᵢ⁰ ⊕ Δ`, Free-XOR). The evaluator holds
//! exactly one label per bit, decrypts its row, and recombines
//! `Σ 2ⁱ rᵢⱼ = aⱼ·x + bⱼ`.
//!
//! # Hash count (fixed by the protocol, not the implementation)
//!
//! Masking one `F_p` element needs `B = ⌈lg p / λ⌉ = 2` blocks. One row of one
//! bit covers an `S`-vector: `S·B` hash calls. The garbler builds **both** rows
//! per bit (it does not know `xᵢ`); the evaluator opens **one**:
//!
//! ```text
//! garble = 2 · n · S · B = 4nS     eval = n · S · B = 2nS
//! ```

use crate::crypto::{Block, expand};
use rand::Rng;

/// Number of bits `x` is decomposed into (the s-aff workload's `N=256`).
///
/// The BN254 prime is 254-bit, so the top two bit-positions are always zero, but
/// the garbler still garbles all `N_BITS` positions since it cannot know `x`.
pub const N_BITS: usize = 256;

/// Bytes in the canonical encoding of an `F_p` element.
pub const FE_BYTES: usize = 32;

/// 64-bit limbs per field element (`FE_BYTES / 8`).
const FE_LIMBS: usize = FE_BYTES / 8;

/// One CCRH/AES block (λ = 128 bits) in bytes.
const BLOCK_BYTES: usize = 16;

/// Hash blocks needed to one-time-pad a single field element:
/// `⌈FE_BYTES / BLOCK_BYTES⌉ = 2` (this is the `B = ⌈lg p / λ⌉` factor).
pub const BLOCKS_PER_FE: usize = FE_BYTES.div_ceil(BLOCK_BYTES);

/// BN254 base field modulus `q`, little-endian 64-bit limbs.
const P: [u64; 4] = [
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

// ---- F_p (BN254 base field) ----

/// An element of `F_p` (BN254 base field), normalized to `value < p`, stored as
/// little-endian 64-bit limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp([u64; 4]);

/// 256-bit add with carry-out (no `u128`).
#[inline]
fn adc(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let mut r = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        r[i] = s2;
        carry = (c1 as u64) | (c2 as u64);
    }
    (r, carry)
}

/// 256-bit subtract; returns `(a - b mod 2²⁵⁶, borrow)` (no `i128`).
#[inline]
fn sbb(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let mut r = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        r[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    (r, borrow)
}

/// 256-bit left shift by one; returns `(2a mod 2²⁵⁶, carry-out)`.
#[inline]
fn shl1(a: [u64; 4]) -> ([u64; 4], u64) {
    let mut r = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        r[i] = (a[i] << 1) | carry;
        carry = a[i] >> 63;
    }
    (r, carry)
}

/// Branchless conditional subtract: returns `v` if `v < p`, else `v - p`. Used
/// for every modular reduction so the hot field loops never branch on
/// data-dependent (≈50/50) comparisons.
#[inline]
fn csub_p(v: [u64; 4]) -> [u64; 4] {
    let (d, borrow) = sbb(v, P); // borrow == 1 iff v < p
    let keep = 0u64.wrapping_sub(borrow); // all-ones iff v < p (keep v), else 0 (take v-p)
    std::array::from_fn(|k| (v[k] & keep) | (d[k] & !keep))
}

impl Fp {
    /// Additive identity.
    pub fn zero() -> Self {
        Fp([0; 4])
    }

    /// Embed a `u64` (always `< p`).
    pub fn from_u64(v: u64) -> Self {
        Fp([v, 0, 0, 0])
    }

    /// `self + other mod p`.
    // Inherent field op, deliberately not `std::ops::Add` (keeps `Fp` operator-free).
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn add(self, other: Fp) -> Fp {
        // a, b < p < 2²⁵⁴, so a+b < 2²⁵⁵ — never carries out of 4 limbs, and at
        // most one conditional subtraction normalizes.
        let (s, carry) = adc(self.0, other.0);
        debug_assert_eq!(carry, 0, "Fp::add operands were not normalized");
        Fp(csub_p(s))
    }

    /// `self - other mod p` (branchless: add `p` back iff the subtract borrowed).
    // Inherent field op, deliberately not `std::ops::Sub` (see `add`).
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn sub(self, other: Fp) -> Fp {
        let (d, borrow) = sbb(self.0, other.0);
        let add_p = 0u64.wrapping_sub(borrow); // all-ones iff a < b
        let (r, _) = adc(d, [P[0] & add_p, P[1] & add_p, P[2] & add_p, P[3] & add_p]);
        Fp(r)
    }

    /// `2·self mod p`, via a 1-bit shift + one branchless conditional subtract
    /// (`2·self < 2p`). This is the only "multiplication" the workload needs.
    #[inline]
    pub fn double(self) -> Fp {
        let (s, _) = shl1(self.0); // self < 2²⁵⁴ ⇒ no carry out
        Fp(csub_p(s))
    }

    /// `2·self + r mod p` — the fused Horner step. `2·self + r < 3p < 2²⁵⁶`, so
    /// it reduces with two branchless conditional subtracts and a single add,
    /// instead of normalizing the double and the add separately.
    #[inline]
    pub fn double_add(self, r: Fp) -> Fp {
        let (d, _) = shl1(self.0); // 2·self < 2²⁵⁵, no carry
        let (t, _) = adc(d, r.0); // < 3p < 2²⁵⁶, no carry
        Fp(csub_p(csub_p(t)))
    }

    /// Sampled field element. Draws 254 random bits (so the value is
    /// `< 2²⁵⁴ < 2p`) and conditionally subtracts `p` once. Slightly
    /// non-uniform — immaterial for a correctness/runtime baseline; a production
    /// sampler would draw wider and reduce, or smudge.
    #[inline]
    pub fn random(rng: &mut impl Rng) -> Fp {
        let v = [
            rng.random::<u64>(),
            rng.random::<u64>(),
            rng.random::<u64>(),
            rng.random::<u64>() & ((1u64 << 62) - 1),
        ];
        Fp(csub_p(v))
    }

    /// Canonical 32-byte little-endian encoding.
    pub fn to_bytes(self) -> [u8; FE_BYTES] {
        let mut out = [0u8; FE_BYTES];
        for (i, limb) in self.0.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// Decode a canonical 32-byte little-endian encoding (must be `< p`).
    pub fn from_bytes(bytes: &[u8]) -> Fp {
        debug_assert_eq!(bytes.len(), FE_BYTES);
        let mut v = [0u64; 4];
        for (i, limb) in v.iter_mut().enumerate() {
            *limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        debug_assert!(sbb(v, P).1 == 1, "from_bytes: value not reduced mod p"); // borrow ⇒ v < p
        Fp(v)
    }

    /// Bit `i` of the canonical integer representation.
    #[inline]
    fn bit(self, i: usize) -> bool {
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Reduce a 256-bit value (a hash pad) into `F_p` the same way [`random`]
    /// does: mask to 254 bits, one conditional subtract. Used by GRR to derive a
    /// share / mask directly from `H(label)`. Slightly non-uniform — immaterial
    /// for a correctness/runtime baseline.
    ///
    /// [`random`]: Fp::random
    #[inline]
    fn reduce_limbs(mut v: [u64; 4]) -> Fp {
        v[3] &= (1u64 << 62) - 1;
        Fp(csub_p(v))
    }
}

/// `Σ 2ⁱ rᵢ mod p` (Horner, high index to low). Pure field arithmetic — the
/// recombination is linear and costs zero hash calls.
pub fn recombine(r: &[Fp]) -> Fp {
    let mut acc = Fp::zero();
    for &ri in r.iter().rev() {
        acc = acc.double_add(ri);
    }
    acc
}

// ---- Hash accounting ----

/// Tally of CCRH (AES-block / λ-bit) invocations, split by party. Counts are
/// measured by instrumenting every [`expand`] call, not derived from a formula.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HashLedger {
    /// Hash calls made by the garbler.
    pub garble_calls: usize,
    /// Hash calls made by the evaluator.
    pub eval_calls: usize,
}

/// [`expand`] into `out`, tallying the number of 16-byte (λ-bit) blocks emitted.
#[inline]
fn expand_counted(seed: Block, nonce: u64, out: &mut [u8], counter: &mut usize) {
    expand(seed, nonce, out);
    *counter += out.len().div_ceil(BLOCK_BYTES);
}

/// View a `u64` slice as a mutable byte slice for AES-CTR fill / XOR.
#[inline]
fn u64_bytes_mut(s: &mut [u64]) -> &mut [u8] {
    // SAFETY: `u64` has no invalid bit patterns and is more strictly aligned
    // than `u8`; the returned slice covers exactly the same `len*8` bytes and
    // borrows `s` for its lifetime. Byte order is whatever the host uses, which
    // is consistent across garble and eval (ciphertexts are never compared
    // across machines).
    unsafe {
        core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(s))
    }
}

/// XOR two 128-bit blocks.
#[inline]
fn xor_block(a: Block, b: Block) -> Block {
    std::array::from_fn(|k| a[k] ^ b[k])
}

/// Allocate `n` u64 words in a single contiguous block, **without**
/// zero-initialization. Used for the ciphertext banks: the whole
/// `N_BITS · row_words` size is known from `S` and reserved up front (one
/// allocation, no incremental growth), and every word is written by `expand`
/// before it is ever read — so the multi-GB zero-fill `vec![0u64; n]` would do
/// is pure waste at large `S`.
// uninit is deliberate — `garble` writes every word via `expand` before any read (SAFETY below).
#[allow(clippy::uninit_vec)]
fn alloc_uninit_u64(n: usize) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    // SAFETY: `u64` is `Copy` with no invalid bit patterns; capacity is `n`, and
    // `garble` fills every word via `expand` before any read (the in-place XOR,
    // and later `eval`), so no uninitialized word is ever read.
    unsafe {
        v.set_len(n);
    }
    v
}

/// Closed-form hash counts for `S` maps: `garble = 4nS`, `eval = 2nS` (with
/// `B = BLOCKS_PER_FE = 2`). Lets callers project costs without running the
/// (24 MiB-communication) garbling.
pub fn cost_model(s: usize) -> HashLedger {
    let per_row = s * BLOCKS_PER_FE; // S·B hash calls for one row of one bit
    HashLedger {
        garble_calls: N_BITS * 2 * per_row,
        eval_calls: N_BITS * per_row,
    }
}

// ---- Garbling ----

/// The garbler's output under **garbled-row-reduction (GRR)**: only the
/// transmitted ciphertexts are materialized.
///
/// The `xᵢ=0` rows for bits `1..n` are *free* — their share is defined as
/// `bᵢ := reduce(H(Lᵢ⁰))`, so the masked
/// value is 0 and the row need not be stored or sent. Only the `n` row-1
/// ciphertexts (the `xᵢ=1` selections) plus the single non-free row-0 (bit 0,
/// whose share is forced to meet the `Σ 2ⁱ bᵢ = b` constraint) are kept. This
/// halves both communication and the garbler's memory traffic versus sending
/// both rows, and removes share sampling entirely (shares come from `H`).
///
/// Ciphertexts are XOR-masked over the canonical encoding and stored as `u64`
/// limbs in one contiguous, exactly-sized bank (reserved from `S` up front).
#[derive(Clone, Debug)]
pub struct Garbling {
    /// Number of affine maps.
    pub s: usize,
    /// `u64` words per ciphertext row (`= s · FE_LIMBS`).
    row_words: usize,
    /// Row-1 ciphertext for every bit, one contiguous `N_BITS · row_words` bank;
    /// row `i` is `ct1[i*row_words ..]`, holding `(a + bᵢ) ⊕ H(Lᵢ¹)`.
    ct1: Vec<u64>,
    /// The one non-free row-0 ciphertext (bit 0): `b₀ ⊕ H(L₀⁰)`.
    ct0_0: Vec<u64>,
    /// `Lᵢ⁰` (the `xᵢ=0` label) for each of the `N_BITS` input bits.
    labels0: Vec<Block>,
    /// Free-XOR global offset: `Lᵢ¹ = Lᵢ⁰ ⊕ Δ`.
    delta: Block,
}

impl Garbling {
    /// What sending **both** rows would cost, in bytes (`2nS` field elements) —
    /// reported for reference; GRR does not materialize this.
    pub fn comm_bytes(&self) -> usize {
        2 * N_BITS * self.s * FE_BYTES
    }

    /// Bytes actually transmitted/materialized under GRR: `n` row-1 banks plus
    /// the single non-free row-0.
    pub fn comm_bytes_grr(&self) -> usize {
        8 * (self.ct1.len() + self.ct0_0.len())
    }
}

/// Garble the `S` affine maps `aⱼ·x + bⱼ`. `a` and `b` must have equal length.
///
/// Tallies the garbler's hash calls into `ledger.garble_calls`. Single pass over
/// the bits: each share is generated once, used for both rows and folded into a
/// per-column Horner accumulator, so no `n·S` share buffer is materialized.
pub fn garble(a: &[Fp], b: &[Fp], rng: &mut impl Rng, ledger: &mut HashLedger) -> Garbling {
    let s = a.len();
    assert_eq!(s, b.len(), "a and b must have equal length");
    let row_words = s * FE_LIMBS;

    let delta: Block = rng.random();
    let labels0: Vec<Block> = (0..N_BITS).map(|_| rng.random()).collect();

    // One contiguous bank for the n transmitted row-1 ciphertexts, plus the one
    // non-free row-0; both reserved exactly from S, uninitialized (every word is
    // written by `expand` before any read).
    let mut ct1 = alloc_uninit_u64(N_BITS * row_words);
    let mut ct0_0 = alloc_uninit_u64(row_words);
    let mut scratch0 = alloc_uninit_u64(row_words); // raw H(Lᵢ⁰), reused each bit

    // Horner accumulator for Σ_{i≥1} 2^{i-1} bᵢⱼ, with bᵢ := reduce(H(Lᵢ⁰)).
    let mut acc = vec![Fp::zero(); s];

    // Bits n-1 .. 1: the xᵢ=0 row is free (bᵢ = pad0ᵢ); only row-1 is stored.
    for i in (1..N_BITS).rev() {
        let l0 = labels0[i];
        let l1 = xor_block(l0, delta);
        let off = i * row_words;
        expand_counted(
            l0,
            i as u64,
            u64_bytes_mut(&mut scratch0),
            &mut ledger.garble_calls,
        );
        // Expand H(Lᵢ¹) straight into the bank slot; reduce it in place to pad1,
        // then overwrite with the computed ciphertext.
        expand_counted(
            l1,
            i as u64,
            u64_bytes_mut(&mut ct1[off..off + row_words]),
            &mut ledger.garble_calls,
        );
        let row1 = &mut ct1[off..off + row_words];
        for (((c1, p0), accj), &aj) in row1
            .chunks_exact_mut(FE_LIMBS)
            .zip(scratch0.chunks_exact(FE_LIMBS))
            .zip(acc.iter_mut())
            .zip(a.iter())
        {
            let bij = Fp::reduce_limbs([p0[0], p0[1], p0[2], p0[3]]); // bᵢⱼ = reduce(H(Lᵢ⁰))
            *accj = accj.double_add(bij);
            let ab = aj.add(bij);
            // parallel index into c1 and ab.0 (raw pad → ciphertext, in place)
            #[allow(clippy::needless_range_loop)]
            for k in 0..FE_LIMBS {
                c1[k] ^= ab.0[k]; // (a + bᵢ) ⊕ H(Lᵢ¹)  (c1 holds raw H(Lᵢ¹))
            }
        }
    }

    // Bit 0 (weight 1): the share is forced so Σ_i 2ⁱ bᵢⱼ = bⱼ; both rows stored.
    {
        let l0 = labels0[0];
        let l1 = xor_block(l0, delta);
        expand_counted(
            l0,
            0,
            u64_bytes_mut(&mut scratch0),
            &mut ledger.garble_calls,
        );
        expand_counted(
            l1,
            0,
            u64_bytes_mut(&mut ct1[0..row_words]),
            &mut ledger.garble_calls,
        );
        let row1 = &mut ct1[0..row_words];
        for ((((c1, c0), p0), &accj), (&aj, &bj)) in row1
            .chunks_exact_mut(FE_LIMBS)
            .zip(ct0_0.chunks_exact_mut(FE_LIMBS))
            .zip(scratch0.chunks_exact(FE_LIMBS))
            .zip(acc.iter())
            .zip(a.iter().zip(b.iter()))
        {
            let rj = accj.double(); // Σ_{i≥1} 2ⁱ bᵢⱼ
            let b0 = bj.sub(rj); // b₀ⱼ
            let ab0 = aj.add(b0);
            for k in 0..FE_LIMBS {
                c0[k] = b0.0[k] ^ p0[k]; // row-0: b₀ ⊕ H(L₀⁰)  (p0 = raw H(L₀⁰))
                c1[k] ^= ab0.0[k]; // row-1: (a + b₀) ⊕ H(L₀¹)  (c1 = raw H(L₀¹))
            }
        }
    }

    Garbling {
        s,
        row_words,
        ct1,
        ct0_0,
        labels0,
        delta,
    }
}

/// Select the evaluator's active input labels for `x` (bit `i` of `x`). The
/// evaluator obtains these via OT / Lamport signature in a real protocol.
pub fn encode(g: &Garbling, x: Fp) -> Vec<Block> {
    (0..N_BITS)
        .map(|i| {
            let l0 = g.labels0[i];
            if x.bit(i) { xor_block(l0, g.delta) } else { l0 }
        })
        .collect()
}

/// Evaluate: decrypt one row per bit and recombine to recover `aⱼ·x + bⱼ` for
/// every map. `x` (the evaluator's own input) selects which row each label opens.
///
/// Tallies the evaluator's hash calls into `ledger.eval_calls`. Single pass,
/// high bit to low, folding each decrypted row straight into a per-column Horner
/// accumulator (`O(S)` state, no `n·S` materialization).
pub fn eval(g: &Garbling, active: &[Block], x: Fp, ledger: &mut HashLedger) -> Vec<Fp> {
    let s = g.s;
    let row_words = g.row_words;
    let mut acc = vec![Fp::zero(); s];
    let mut pad_raw = vec![0u64; row_words]; // raw H(active[i]), reused each bit

    for i in (0..N_BITS).rev() {
        let off = i * row_words;
        let xi = x.bit(i);
        expand_counted(
            active[i],
            i as u64,
            u64_bytes_mut(&mut pad_raw),
            &mut ledger.eval_calls,
        );
        // The stored ciphertext backing this row (if any): row-1 when xᵢ=1, the
        // non-free row-0 when (i=0, x₀=0); otherwise the row is free.
        let ct: Option<&[u64]> = if xi {
            Some(&g.ct1[off..off + row_words])
        } else if i == 0 {
            Some(&g.ct0_0)
        } else {
            None
        };
        match ct {
            // Stored row: rᵢⱼ = ctⱼ ⊕ pad — XOR-unmask to the canonical value (< p).
            Some(ct) => {
                for ((cj, pj), accj) in ct
                    .chunks_exact(FE_LIMBS)
                    .zip(pad_raw.chunks_exact(FE_LIMBS))
                    .zip(acc.iter_mut())
                {
                    let r = Fp([cj[0] ^ pj[0], cj[1] ^ pj[1], cj[2] ^ pj[2], cj[3] ^ pj[3]]);
                    *accj = accj.double_add(r);
                }
            }
            // Free row (xᵢ=0, i≥1): rᵢⱼ = bᵢⱼ = reduce(H(Lᵢ⁰)) — no ciphertext read.
            None => {
                for (pj, accj) in pad_raw.chunks_exact(FE_LIMBS).zip(acc.iter_mut()) {
                    let r = Fp::reduce_limbs([pj[0], pj[1], pj[2], pj[3]]);
                    *accj = accj.double_add(r);
                }
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Fast non-cryptographic PRG for benchmarking (so RNG cost does not
    /// dominate the measurement of garbling work). Stands in for the garbler's
    /// local PRG / random-oracle expansion of share randomness.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            SplitMix64 { state: seed }
        }
    }

    impl rand::RngCore for SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            let mut i = 0;
            while i < dst.len() {
                let r = self.next_u64().to_le_bytes();
                let n = (dst.len() - i).min(8);
                dst[i..i + n].copy_from_slice(&r[..n]);
                i += n;
            }
        }
    }

    /// Plaintext oracle: `a · x mod p` via textbook double-and-add. This is the
    /// only general field multiplication anywhere in the module, and it exists
    /// purely to check correctness — it is never on the garble/eval path.
    fn plaintext_mul(a: Fp, x: Fp) -> Fp {
        let mut acc = Fp::zero();
        for i in (0..256).rev() {
            acc = acc.double();
            if x.bit(i) {
                acc = acc.add(a);
            }
        }
        acc
    }

    // ---- F_p sanity ----

    #[test]
    fn test_fp_add_sub_wraparound() {
        let one = Fp::from_u64(1);
        let p_minus_1 = Fp::zero().sub(one); // 0 - 1 = p - 1
        assert_eq!(p_minus_1.add(one), Fp::zero(), "(p-1) + 1 = 0");
        let mut want = P;
        want[0] -= 1;
        assert_eq!(p_minus_1, Fp(want));
    }

    #[test]
    fn test_fp_double_and_double_add() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let x = Fp::random(&mut rng);
            let r = Fp::random(&mut rng);
            assert_eq!(x.double(), x.add(x), "double must equal x+x");
            assert_eq!(
                x.double_add(r),
                x.double().add(r),
                "double_add must equal 2x+r"
            );
        }
    }

    #[test]
    fn test_plaintext_mul_oracle() {
        assert_eq!(
            plaintext_mul(Fp::from_u64(7), Fp::from_u64(11)),
            Fp::from_u64(77)
        );
        let x = Fp::zero().sub(Fp::from_u64(3)); // p - 3
        assert_eq!(plaintext_mul(Fp::from_u64(2), x), x.double());
        let pm1 = Fp::zero().sub(Fp::from_u64(1));
        assert_eq!(plaintext_mul(pm1, pm1), Fp::from_u64(1)); // (p-1)² = 1
    }

    #[test]
    fn test_recombine_is_weighted_sum() {
        let mut r = vec![Fp::zero(); N_BITS];
        r[0] = Fp::from_u64(3);
        r[1] = Fp::from_u64(1);
        assert_eq!(recombine(&r), Fp::from_u64(5)); // 3 + 2·1
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x = Fp::random(&mut rng);
            assert_eq!(Fp::from_bytes(&x.to_bytes()), x);
        }
    }

    // ---- End-to-end correctness (self-contained) ----

    /// Garble → encode → eval must equal the plaintext `aⱼ·x + bⱼ`, checked
    /// against the independent double-and-add oracle.
    #[test]
    fn test_garble_eval_matches_plaintext() {
        let mut rng = rand::rng();
        let s = 16;
        for _ in 0..8 {
            let a: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
            let b: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
            let x = Fp::random(&mut rng);

            let mut ledger = HashLedger::default();
            let g = garble(&a, &b, &mut rng, &mut ledger);
            let active = encode(&g, x);
            let out = eval(&g, &active, x, &mut ledger);

            for j in 0..s {
                assert_eq!(
                    out[j],
                    plaintext_mul(a[j], x).add(b[j]),
                    "map {j}: a·x+b mismatch"
                );
            }
        }
    }

    /// A different input must produce a (correctly) different output.
    #[test]
    fn test_distinct_inputs_give_distinct_outputs() {
        let mut rng = rand::rng();
        let s = 4;
        let a: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
        let b: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
        let x = Fp::random(&mut rng);
        let x2 = x.add(Fp::from_u64(1));

        let mut ledger = HashLedger::default();
        let g = garble(&a, &b, &mut rng, &mut ledger);

        let out_x = eval(&g, &encode(&g, x), x, &mut ledger);
        let out_x2 = eval(&g, &encode(&g, x2), x2, &mut ledger);
        for j in 0..s {
            assert_eq!(out_x[j], plaintext_mul(a[j], x).add(b[j]), "map {j} at x");
            assert_eq!(
                out_x2[j],
                plaintext_mul(a[j], x2).add(b[j]),
                "map {j} at x2"
            );
        }
        assert!(
            (0..s).any(|j| out_x[j] != out_x2[j]),
            "distinct inputs collapsed to the same output"
        );
    }

    // ---- Measured hash count vs the CRT one-hot-scaling s-aff path ----

    /// Measures the real garble/eval hash calls at the s-aff workload (S=1536)
    /// and asserts they equal the `4nS / 2nS` closed form.
    #[test]
    fn test_hash_count_matches_formula() {
        let mut rng = SplitMix64::new(0xC0FFEE);
        let s = 1536;
        let a: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
        let b: Vec<Fp> = (0..s).map(|_| Fp::random(&mut rng)).collect();
        let x = Fp::random(&mut rng);

        let mut ledger = HashLedger::default();
        let g = garble(&a, &b, &mut rng, &mut ledger);
        let active = encode(&g, x);
        let _ = eval(&g, &active, x, &mut ledger);

        assert_eq!(
            ledger,
            cost_model(s),
            "measured hash count must match 4nS / 2nS"
        );
        assert_eq!(ledger.garble_calls, 2 * N_BITS * s * BLOCKS_PER_FE);
        assert_eq!(ledger.eval_calls, N_BITS * s * BLOCKS_PER_FE);
        assert_eq!(g.comm_bytes(), 2 * N_BITS * s * FE_BYTES); // both-rows reference
        assert_eq!(g.comm_bytes_grr(), (N_BITS + 1) * s * FE_BYTES); // GRR: n row-1 + 1 row-0
    }

    // ---- Runtime benchmark (single-threaded baseline) ----

    /// Measures wall-clock garble/eval runtime at the s-aff workload, the raw
    /// CCRH floor, and overhead above it. Single-threaded, to compare against
    /// the one-hot-scaling s-aff pipeline. Run with:
    /// `cargo test -r bitdecomp::tests::bench_runtime -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_runtime() {
        let s = 1536; // s-aff workload: N=256, S=6·256
        let mut prng = SplitMix64::new(0x0DDC0FFEE0DDF00D);
        let a: Vec<Fp> = (0..s).map(|_| Fp::random(&mut prng)).collect();
        let b: Vec<Fp> = (0..s).map(|_| Fp::random(&mut prng)).collect();
        let x = Fp::random(&mut prng);

        // Warm up allocator + AES round keys.
        let mut warm = HashLedger::default();
        let g0 = garble(&a, &b, &mut prng, &mut warm);
        let active0 = encode(&g0, x);
        black_box(eval(&g0, &active0, x, &mut warm));

        let iters = 50usize;
        let warmup = 5usize;
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        // --- garble (median of per-iter samples; robust to allocation outliers) ---
        let mut last = None;
        let mut g_samples = Vec::with_capacity(iters);
        for it in 0..warmup + iters {
            let mut l = HashLedger::default();
            let t = Instant::now();
            let g = garble(&a, &b, &mut prng, &mut l);
            let dt = t.elapsed().as_nanos() as f64;
            black_box(&g);
            if it >= warmup {
                g_samples.push(dt);
            }
            last = Some(g);
        }
        let garble_ns = median(g_samples);
        let g = last.unwrap();
        let active = encode(&g, x);

        // --- eval (median of per-iter samples) ---
        let mut e_samples = Vec::with_capacity(iters);
        for it in 0..warmup + iters {
            let mut l = HashLedger::default();
            let t = Instant::now();
            black_box(eval(&g, &active, x, &mut l));
            let dt = t.elapsed().as_nanos() as f64;
            if it >= warmup {
                e_samples.push(dt);
            }
        }
        let eval_ns = median(e_samples);

        // --- raw CCRH floor: the same total blocks, no field/XOR work ---
        let model = cost_model(s);
        let g_blocks = model.garble_calls as f64;
        let e_blocks = model.eval_calls as f64;
        let chunk = s * FE_LIMBS; // one row buffer
        let chunk_blocks = (chunk * 8 / BLOCK_BYTES) as u64; // = 2·s
        let total_blocks = (model.garble_calls + model.eval_calls) as u64;
        let reps = (total_blocks / chunk_blocks).max(1);
        let mut buf = vec![0u64; chunk];
        let t = Instant::now();
        for r in 0..reps {
            expand([r as u8; 16], r, u64_bytes_mut(&mut buf));
            black_box(&buf);
        }
        let raw_ns = t.elapsed().as_nanos() as f64;
        let raw_per_block = raw_ns / (reps * chunk_blocks) as f64;

        let g_per_block = garble_ns / g_blocks;
        let e_per_block = eval_ns / e_blocks;
        eprintln!(
            "\n== bit-decomposition a·x+b runtime  (N_BITS={N_BITS}, S={s}, λ=128, single-thread, {iters} iters) ==\n\
             garble : {:>7.2} ms   ({:.2}M blocks, {:.2} ns/block)\n\
             eval   : {:>7.2} ms   ({:.2}M blocks, {:.2} ns/block)\n\
             total  : {:>7.2} ms   ({:.2}M blocks)\n\
             raw CCRH floor      : {:.2} ns/block\n\
             overhead vs floor   : garble {:.2}x, eval {:.2}x\n\
             comm                : {:.1} MiB (both rows) | {:.1} MiB w/ GRR\n\
             ---- reference: one-hot CRT s-aff path, same a·x+b ----\n\
             hash calls : ~2.3M body / ~5.8M full pipeline  (bit-decomp here: {:.2}M)\n\
             => compute baseline; field work is only add/sub/doubling (no mul).\n\
                Compare total ms against the one-hot CRT s-aff bench.\n",
            garble_ns / 1e6,
            g_blocks / 1e6,
            g_per_block,
            eval_ns / 1e6,
            e_blocks / 1e6,
            e_per_block,
            (garble_ns + eval_ns) / 1e6,
            (g_blocks + e_blocks) / 1e6,
            raw_per_block,
            g_per_block / raw_per_block,
            e_per_block / raw_per_block,
            g.comm_bytes() as f64 / (1024.0 * 1024.0),
            g.comm_bytes_grr() as f64 / (1024.0 * 1024.0),
            (g_blocks + e_blocks) / 1e6,
        );
    }
}
