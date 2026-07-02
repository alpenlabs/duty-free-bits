//! The shared machinery steps 1 (chunk) and 2 (extract) build on, over bare
//! label words (Z₂ = boolean label `[u64; 2]`, width-`l` = arithmetic label
//! `[u32; λ]` lanes):
//!
//! * **growing a one-hot one bit at a time** ([`tree_garble`] on the garbler
//!   side, one one-hot scaling per bit; the evaluator resolves the one open
//!   slot per level through that scaling), and
//! * **upcasting every leaf** ([`hash_cast`]) whose weighted sums
//!   ([`peel_chain`]) give both the extracted bits and the root scaling that
//!   delivers the active leaf's upcast value.
//!
//! The garbler hashes EVERY slot (x-blind); the evaluator, knowing `x`, hashes
//! only closed slots. Lane loops run unmasked mod 2^32 (truncation to Z_{2^l}
//! commutes) with masks applied only at the boundaries that make a wire
//! canonical.

use crate::crypto::expand;
use crate::gc::Cost;
use crate::hash;
use crate::label::{LAMBDA, Label};


/// Packed words of a boolean label (the bare-word working type).
pub(crate) type Z2 = [u64; 2];

/// One coordinate per u32 lane of a width-`l` arithmetic label (2 ≤ l ≤ 32).
pub(crate) type Wide = [u32; LAMBDA];

#[inline]
pub(crate) fn z2_of(l: &Label) -> Z2 {
    debug_assert_eq!(l.modulus(), 2);
    let raw = l.raw_bits();
    [raw[0], raw[1]]
}

#[inline]
pub(crate) fn z2_label(w: Z2) -> Label {
    Label::from_raw_bits(vec![w[0], w[1]], 2)
}

#[inline]
pub(crate) fn wide_of(l: &Label) -> Wide {
    l.lanes().try_into().expect("lanes are LAMBDA wide")
}

#[inline]
pub(crate) fn wide_label(w: &Wide, modulus: u64) -> Label {
    let mask = (modulus - 1) as u32;
    Label::from_lanes(w.iter().map(|&l| l & mask).collect(), modulus)
}

/// `dst += src` lane-wise, mod 2^32.
///
/// The lane ops run UNMASKED: wrapping u32 arithmetic is a ring homomorphism
/// onto Z_{2^l} under truncation, so the `& (2^l − 1)` is applied only at the
/// boundaries that make a wire canonical ([`wide_label`], the active-leaf
/// upcast emission, and the Z₂ packing of bits below the preserved width) —
/// not on every op.
#[inline]
pub(crate) fn wide_add(dst: &mut Wide, src: &Wide) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_add(*s);
    }
}

/// `dst −= src` lane-wise, mod 2^32 (see [`wide_add`] for masking).
#[inline]
pub(crate) fn wide_sub(dst: &mut Wide, src: &Wide) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_sub(*s);
    }
}

/// `dst += c·src` lane-wise, mod 2^32 (see [`wide_add`] for masking).
#[inline]
pub(crate) fn wide_madd(dst: &mut Wide, c: u32, src: &Wide) {
    if c == 0 {
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_add(c.wrapping_mul(*s));
    }
}

/// Pack bit `bit` of every lane into a boolean label's words.
#[inline]
pub(crate) fn pack_lane_bit(lanes: &Wide, bit: u32) -> Z2 {
    let mut out = [0u64; 2];
    for (w, chunk) in lanes.chunks_exact(64).enumerate() {
        let mut acc = 0u64;
        for (i, &l) in chunk.iter().enumerate() {
            acc |= (((l >> bit) & 1) as u64) << i;
        }
        out[w] = acc;
    }
    out
}

/// CCRH pad of one width-`l` upcast scaling, unpacked to lanes: the
/// arithmetic-label Z_{2^l} analogue of [`hash::hash_z2`] for wider payloads on
/// bare words. `nonce` is a solo-domain id (bit 63 clear), globally fresh.
pub(crate) fn hash_cast(ctrl: &Z2, nonce: u64, l: u32, out: &mut Wide) {
    debug_assert!(nonce < (1u64 << 63), "solo nonce uses the bulk-domain bit");
    debug_assert!((2..=32).contains(&l));
    let mut seed = [0u8; 16];
    seed[0..8].copy_from_slice(&ctrl[0].to_le_bytes());
    seed[8..16].copy_from_slice(&ctrl[1].to_le_bytes());
    let words = (LAMBDA * l as usize).div_ceil(64);
    // 16 tail bytes keep the NEON gather's last 16-byte load in bounds.
    let mut buf = [0u8; 512 + 16];
    expand(seed, nonce, &mut buf[..words * 8]);
    #[cfg(target_arch = "aarch64")]
    if neon_unpack_ok(l as usize) {
        unpack_even_k_neon(&buf, l as usize, out);
        return;
    }
    unpack_generic(&buf[..words * 8], l as usize, out);
}

#[inline]
fn lane_mask(k: u32) -> u32 {
    if k >= 32 { u32::MAX } else { (1u32 << k) - 1 }
}

/// Scalar unpack of LAMBDA packed `k`-bit coordinates from little-endian
/// bytes into u32 lanes (branchless two-word windows; cf. label.rs).
fn unpack_generic(bytes: &[u8], k: usize, dst: &mut Wide) {
    let mut padded = [0u64; LAMBDA * 32 / 64 + 1];
    for (w, slot) in padded.iter_mut().enumerate().take(bytes.len() / 8) {
        *slot = u64::from_le_bytes(bytes[w * 8..(w + 1) * 8].try_into().unwrap());
    }
    let mask = lane_mask(k as u32) as u64;
    for (i, lane) in dst.iter_mut().enumerate() {
        let bit = i * k;
        let window = (padded[bit >> 6] as u128) | ((padded[(bit >> 6) + 1] as u128) << 64);
        *lane = ((window >> (bit & 63)) as u64 & mask) as u32;
    }
}

/// True iff [`unpack_even_k_neon`] handles width `k`: even (group bases stay
/// byte-aligned) and every lane's `shift + k` fits the 32-bit gather window
/// (this excludes exactly k = 30, whose lane-1 shift of 6 overflows it).
#[cfg(target_arch = "aarch64")]
fn neon_unpack_ok(k: usize) -> bool {
    if !k.is_multiple_of(2) || !(2..=32).contains(&k) {
        return false;
    }
    (0..4).all(|j| ((j * k) & 7) + k <= 32)
}

/// NEON unpack for even `k` (2 ≤ k ≤ 32): per 4-lane group, one unaligned
/// 16-byte load at the group's (byte-aligned) base, a TBL byte-gather of each
/// lane's 4-byte window, per-lane right shifts and a mask.
///
/// Even k makes every group base byte-aligned (`4k ≡ 0 (mod 8)`), so the
/// gather indices and shifts are group-invariant. `scratch` must be at least
/// `(124·k)/8 + 16` bytes (the 512-byte hash scratch always is for k ≤ 32);
/// lane 3's window ends at byte 3k/8 + 4 ≤ 16 within each load.
#[cfg(target_arch = "aarch64")]
fn unpack_even_k_neon(scratch: &[u8], k: usize, dst: &mut Wide) {
    use std::arch::aarch64::*;
    debug_assert!(neon_unpack_ok(k));
    assert!(
        scratch.len() >= (124 * k) / 8 + 16,
        "unpack scratch too short"
    );

    // Lane j of a group reads bytes (j·k)/8 .. +4, then shifts by (j·k) % 8.
    let mut idx = [0u8; 16];
    let mut shifts = [0i32; 4];
    for j in 0..4 {
        let bit = j * k;
        for b in 0..4 {
            idx[j * 4 + b] = ((bit >> 3) + b) as u8;
        }
        shifts[j] = -((bit & 7) as i32);
    }
    // SAFETY: NEON is part of the crate's aarch64 baseline (the CCRH core
    // already requires it). All loads are in-bounds: group g loads 16 bytes
    // at (g·4·k)/8 with g ≤ 31, bounded by the assert above.
    unsafe {
        let idx_v = vld1q_u8(idx.as_ptr());
        let shift_v = vld1q_s32(shifts.as_ptr());
        let mask_v = vdupq_n_u32(lane_mask(k as u32));
        for g in 0..(LAMBDA / 4) {
            let base = (g * 4 * k) >> 3;
            let window = vld1q_u8(scratch.as_ptr().add(base));
            let gathered = vreinterpretq_u32_u8(vqtbl1q_u8(window, idx_v));
            let aligned = vshlq_u32(gathered, shift_v);
            let lanes = vandq_u32(aligned, mask_v);
            vst1q_u32(dst.as_mut_ptr().add(g * 4), lanes);
        }
    }
}

/// `bit i` of `sub − res` as a boolean label: the free-recombination step
/// `mod2k(div2k(sub − mod2k(res, k), i), 1)` on lanes. `sub` is width-k
/// lanes, `res` width-l lanes (l ≥ k).
#[inline]
pub(crate) fn peel_bit(sub: &Wide, res: &Wide, i: u32, k: u32) -> Z2 {
    let mask_k = ((1u64 << k) - 1) as u32;
    let mut out = [0u64; 2];
    for (w, (sc, rc)) in sub
        .chunks_exact(64)
        .zip(res.chunks_exact(64))
        .enumerate()
    {
        let mut acc = 0u64;
        for (idx, (&s, &r)) in sc.iter().zip(rc).enumerate() {
            let d = s.wrapping_sub(r) & mask_k;
            acc |= (((d >> i) & 1) as u64) << idx;
        }
        out[w] = acc;
    }
    out
}

/// One grow-plus-upcast stage: `n_bits` one-hot bits, upcasts at width `l`.
pub(crate) fn stage_cost(n_bits: u32, l: u32) -> Cost {
    let n = 1usize << n_bits;
    Cost {
        program_bits: (n_bits as usize - 1 + l as usize) * LAMBDA,
        join_complexity: n_bits as usize - 1 + l as usize,
        hash_count: (n - 2) + l as usize * n,
    }
}

/// Bulk-domain ids growing one one-hot consumes (the garbler hashes every slot
/// of levels 1..k−1; the evaluator indexes the same window by slot position).
#[inline]
pub(crate) fn tree_ids(k: u32) -> u64 {
    (1u64 << k) - 2
}

/// Garbler side of growing one one-hot one bit at a time: level-major, hashes
/// EVERY slot. Returns (leaf masks, per-level y-XOR sums `⊕_j y_m[j]` for
/// levels 1..k−1).
///
/// `lvl1` is the level-1 pair `[X_{b0} ⊕ Δ₂, X_{b0}]` (the affine base
/// one-hot). The scaling ids are `bulk_base + (2^m − 2) + j` for slot `j` of
/// level `m` — position-indexed so the evaluator lands on the same pads.
pub(crate) fn tree_garble(k: u32, lvl1: [Z2; 2], bulk_base: u64) -> (Vec<Z2>, Vec<Z2>) {
    let mut lvl: Vec<Z2> = lvl1.to_vec();
    let mut ysums: Vec<Z2> = Vec::with_capacity(k as usize - 1);
    for m in 1..k {
        let width = 1usize << m;
        let base = bulk_base + ((1u64 << m) - 2);
        let mut next = vec![[0u64; 2]; width * 2];
        let mut ysum = [0u64; 2];
        for j in 0..width {
            let y = hash::hash_z2(&lvl[j], base + j as u64);
            ysum[0] ^= y[0];
            ysum[1] ^= y[1];
            next[j] = [lvl[j][0] ^ y[0], lvl[j][1] ^ y[1]];
            next[j + width] = y;
        }
        ysums.push(ysum);
        lvl = next;
    }
    (lvl, ysums)
}

/// Width-`l` residue bank over a full array of upcast leaves (garbler side,
/// which holds every upcast): halving class sums with the incremental weighted
/// chain `res_{j+1} = res_j + 2^j·H_j`. Returns `(res_1..=res_k, root)`;
/// `res_k` is the upcast functional `Σ p·upcasts[p]`, `root = Σ_p upcasts[p]`.
/// Consumes the upcasts (halved in place).
pub(crate) fn peel_chain(mut acc: Vec<Wide>) -> (Vec<Wide>, Wide) {
    let k = acc.len().trailing_zeros();
    let mut chain: Vec<Wide> = Vec::with_capacity(k as usize);
    // H_j for j = k−1 down to 1, gathered during halving.
    let mut highs: Vec<Wide> = Vec::with_capacity(k as usize);
    while acc.len() > 2 {
        let mid = acc.len() / 2;
        let mut high = [0u32; LAMBDA];
        for d in 0..mid {
            let (lo, hi) = acc.split_at_mut(mid);
            wide_add(&mut high, &hi[d]);
            wide_add(&mut lo[d], &hi[d]);
        }
        acc.truncate(mid);
        highs.push(high);
    }
    // Level 1: res_1 = S_1[1]; H_0-style entry not needed (bit 0 is native).
    let mut res = acc[1];
    chain.push(res);
    // res_{j+1} = res_j + 2^j·H_j, H_j in reverse gather order.
    for (j, high) in (1..k).zip(highs.iter().rev()) {
        let mut r = res;
        wide_madd(&mut r, 1u32 << j, high);
        chain.push(r);
        res = r;
    }
    let mut root = acc[0];
    wide_add(&mut root, &acc[1]);
    (chain, root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// The NEON even-k unpacker must agree with the scalar window loop on
    /// every width it claims (the upcast pads flow through this).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_unpack_even_k_neon_matches_generic() {
        let mut rng = rand::rng();
        let mut scratch = vec![0u8; 512];
        for round in 0..8 {
            rng.fill(&mut scratch[..]);
            for k in (2..=32).step_by(2).filter(|&k| neon_unpack_ok(k)) {
                let bytes = (LAMBDA * k).div_ceil(64) * 8;
                let mut want = [0u32; LAMBDA];
                unpack_generic(&scratch[..bytes], k, &mut want);
                let mut got = [0u32; LAMBDA];
                unpack_even_k_neon(&scratch, k, &mut got);
                assert_eq!(want, got, "k={k}, round={round}");
            }
            assert!(!neon_unpack_ok(30) && !neon_unpack_ok(7));
        }
    }
}
