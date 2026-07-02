//! aarch64-specific intrinsics: AES-NI key schedule + CCRND hash.
//!
//! Lifted from the `gobble` crate (`/Users/nakul/ckt/crates/gobble/src/aarch64/mod.rs`).
//!
//! `ccrnd_with_round_keys` implements the CCRND construction from §5 of
//! <https://eprint.iacr.org/2019/074.pdf>:
//!
//!   H(x, t) = AES_K( σ(x ⊕ public_s ⊕ t) ) ⊕ σ(x ⊕ public_s ⊕ t)
//!
//! where σ is the linear orthomorphism `(L||R) → (L⊕R)||L`.

use std::arch::aarch64::*;
use std::mem::transmute;

mod expand;

/// Pre-expanded AES-128 round keys (11 round keys for the standard schedule).
pub type Aes128RoundKeys = expand::Aes128RoundKeys;

/// AES-128 key expansion.
///
/// # Safety
/// CPU must support `aes` target feature.
#[target_feature(enable = "aes")]
pub unsafe fn expand_aes128_key(key: &[u8; 16]) -> Aes128RoundKeys {
    unsafe { expand::expand_key::<16, 11>(key) }
}

/// XOR of two 128-bit values.
///
/// # Safety
/// CPU must support `neon`.
#[inline]
unsafe fn xor128(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    unsafe { veorq_u8(a, b) }
}

/// Encode a u128 tweak into a 128-bit register.
///
/// # Safety
/// CPU must support `neon` (the result is a `uint8x16_t`).
#[inline]
pub unsafe fn u128_to_block(t: u128) -> uint8x16_t {
    unsafe { transmute::<[u8; 16], uint8x16_t>(t.to_le_bytes()) }
}

/// AES-128 single-block encryption with caller-supplied round keys.
///
/// # Safety
/// CPU must support `aes` and `neon`.
#[target_feature(enable = "aes")]
#[target_feature(enable = "neon")]
unsafe fn aes_encrypt_with_round_keys(
    block: uint8x16_t,
    round_keys: &Aes128RoundKeys,
) -> uint8x16_t {
    let mut state = block;
    for key in round_keys.iter().take(9) {
        state = vaeseq_u8(state, *key);
        state = vaesmcq_u8(state);
    }
    let key9: uint8x16_t = round_keys[9];
    state = vaeseq_u8(state, key9);
    let key10: uint8x16_t = round_keys[10];
    state = veorq_u8(state, key10);
    state
}

/// Linear orthomorphism: `L || R → (L ⊕ R) || L`. From §7.3 of
/// <https://eprint.iacr.org/2019/074.pdf>.
///
/// # Safety
/// CPU must support `neon`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sigma(x: uint8x16_t) -> uint8x16_t {
    let swapped = vextq_u8(x, x, 8);
    let swapped_xor = veorq_u8(x, swapped);
    vextq_u8(swapped_xor, x, 8)
}

/// CCRND hash: `H(x, t) = AES_K(σ(x ⊕ s ⊕ t)) ⊕ σ(x ⊕ s ⊕ t)`.
///
/// # Safety
/// CPU must support `aes` and `neon`. Inputs must be valid 128-bit blocks.
#[inline]
#[target_feature(enable = "aes")]
#[target_feature(enable = "neon")]
pub unsafe fn ccrnd_with_round_keys(
    x: uint8x16_t,
    tweak: uint8x16_t,
    round_keys: &Aes128RoundKeys,
    public_s: uint8x16_t,
) -> uint8x16_t {
    let input = unsafe { xor128(xor128(x, public_s), tweak) };
    let lin_orth_input = unsafe { sigma(input) };
    unsafe {
        xor128(
            aes_encrypt_with_round_keys(lin_orth_input, round_keys),
            lin_orth_input,
        )
    }
}

/// CCRND hash on `N` independent `(x, tweak)` pairs, computed round-major.
///
/// Single-block CCRND is latency-bound: 10 dependent AES rounds at ~2 cycles
/// each leave the core's AES pipes mostly idle. Here the σ inputs are formed
/// for all `N` lanes first, then AES round `r` is applied to every lane before
/// round `r+1`, so independent states fill the pipes. Both loops fully unroll
/// (`N` and the round count are compile-time constants) and the states stay in
/// NEON registers. Each lane is byte-identical to `ccrnd_with_round_keys` on
/// the same `(x, tweak)`.
///
/// # Safety
/// CPU must support `aes` and `neon`. Inputs must be valid 128-bit blocks.
#[target_feature(enable = "aes")]
#[target_feature(enable = "neon")]
pub unsafe fn ccrnd_wide_with_round_keys<const N: usize>(
    xs: [uint8x16_t; N],
    tweaks: [uint8x16_t; N],
    round_keys: &Aes128RoundKeys,
    public_s: uint8x16_t,
) -> [uint8x16_t; N] {
    unsafe {
        let mut lin = [vdupq_n_u8(0); N];
        for i in 0..N {
            lin[i] = sigma(xor128(xor128(xs[i], public_s), tweaks[i]));
        }
        let mut state = lin;
        for key in round_keys.iter().take(9) {
            for s in state.iter_mut() {
                *s = vaesmcq_u8(vaeseq_u8(*s, *key));
            }
        }
        let key9: uint8x16_t = round_keys[9];
        let key10: uint8x16_t = round_keys[10];
        for (s, l) in state.iter_mut().zip(lin) {
            // Final round (no MixColumns), last whitening key, CCRND feed-forward.
            // XOR order differs from the single-block path but XOR is associative
            // and commutative, so the bytes are identical.
            *s = xor128(xor128(vaeseq_u8(*s, key9), key10), l);
        }
        state
    }
}

/// CCRND CTR fill: write whole groups of `W` blocks into `out`, round-major.
///
/// Block `j` (counter `counter + j`) uses tweak `nonce | (counter + j) << 64`
/// and is byte-identical to `ccrnd_with_round_keys` on that tweak. Runs the
/// whole group loop here so the `W` AES states live in NEON registers across
/// calls to the wide kernel (which inlines: same target features) instead of
/// crossing the safe/target-feature ABI boundary once per group. Stops when
/// fewer than `16 * W` bytes remain; returns the number of blocks written.
///
/// # Safety
/// CPU must support `aes` and `neon`.
#[target_feature(enable = "aes")]
#[target_feature(enable = "neon")]
unsafe fn ccrnd_ctr_fill_impl<const W: usize>(
    seed: uint8x16_t,
    nonce: u64,
    counter: u64,
    round_keys: &Aes128RoundKeys,
    public_s: uint8x16_t,
    out: &mut [u8],
) -> usize {
    unsafe {
        let xs = [seed; W];
        let groups = out.len() / (16 * W);
        for g in 0..groups {
            let mut tweaks = [vdupq_n_u8(0); W];
            for (j, t) in tweaks.iter_mut().enumerate() {
                let ctr = counter + (g * W + j) as u64;
                *t = u128_to_block((nonce as u128) | ((ctr as u128) << 64));
            }
            let hs = ccrnd_wide_with_round_keys(xs, tweaks, round_keys, public_s);
            for (j, h) in hs.into_iter().enumerate() {
                // SAFETY: `g < groups` and `j < W` keep the 16-byte store within
                // `out[..groups * 16 * W]`; `vst1q_u8` allows unaligned stores.
                vst1q_u8(out.as_mut_ptr().add((g * W + j) * 16), h);
            }
        }
        groups * W
    }
}

/// Convert little-endian bytes to a 128-bit register.
///
/// # Safety
/// CPU must support `neon` (the result is a `uint8x16_t`).
#[inline]
pub unsafe fn bytes_to_block(bytes: [u8; 16]) -> uint8x16_t {
    unsafe { transmute(bytes) }
}

// --- Backend surface (`[u8; 16]` boundary), shared with the portable backend. ---

/// Pre-expanded AES-128 round keys.
pub type RoundKeys = Aes128RoundKeys;

/// AES-128 key expansion. Safe wrapper — aarch64-apple targets always carry the
/// `aes` feature (ARMv8 crypto extensions).
pub fn expand_key(key: &[u8; 16]) -> RoundKeys {
    unsafe { expand_aes128_key(key) }
}

/// CCRND hash on `[u8; 16]` blocks (NEON/AES-NI internally).
pub fn ccrnd(seed: [u8; 16], tweak: [u8; 16], rk: &RoundKeys, public_s: [u8; 16]) -> [u8; 16] {
    unsafe {
        let out = ccrnd_with_round_keys(
            bytes_to_block(seed),
            bytes_to_block(tweak),
            rk,
            bytes_to_block(public_s),
        );
        transmute::<uint8x16_t, [u8; 16]>(out)
    }
}

/// Interleave width used by `crypto::expand`'s grouped CTR path.
///
/// Measured on Apple M1 (see `bench_ccrnd_single_vs_interleaved`): the grouped
/// kernel sits at the AES-pipe throughput floor, ~1.75 ns/block vs ~4.1 for the
/// old one-block-at-a-time `expand` loop (whose per-block variable-length
/// `copy_from_slice` defeated out-of-order overlap). Widths 4 and 8 measured
/// identical — 4 lanes already saturate both AES pipes — so 4 wins: a smaller
/// group leaves fewer blocks to the single-block tail.
pub const INTERLEAVE: usize = 4;

/// CCRND CTR fill on `[u8; 16]` boundaries, `W` AES states in flight.
///
/// Writes whole groups of `W` blocks of `H(seed, nonce | ctr << 64)` into
/// `out` (counters `counter..`); returns the number of blocks written (a
/// multiple of `W` — the `< 16 * W`-byte tail is left to the caller's
/// single-block loop). Safe wrapper — aarch64-apple targets always carry the
/// `aes`/`neon` features.
pub fn ccrnd_ctr_fill<const W: usize>(
    seed: [u8; 16],
    nonce: u64,
    counter: u64,
    rk: &RoundKeys,
    public_s: [u8; 16],
    out: &mut [u8],
) -> usize {
    unsafe {
        ccrnd_ctr_fill_impl::<W>(
            bytes_to_block(seed),
            nonce,
            counter,
            rk,
            bytes_to_block(public_s),
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccrnd_deterministic() {
        let key = [0x2Bu8; 16];
        let public_s = bytes_to_block_safe([0xDEu8; 16]);
        let round_keys = unsafe { expand_aes128_key(&key) };
        let x = bytes_to_block_safe([0x01u8; 16]);
        let t = bytes_to_block_safe([0x02u8; 16]);
        let a = unsafe { ccrnd_with_round_keys(x, t, &round_keys, public_s) };
        let b = unsafe { ccrnd_with_round_keys(x, t, &round_keys, public_s) };
        let a_bytes: [u8; 16] = unsafe { transmute(a) };
        let b_bytes: [u8; 16] = unsafe { transmute(b) };
        assert_eq!(a_bytes, b_bytes);
    }

    fn bytes_to_block_safe(bytes: [u8; 16]) -> uint8x16_t {
        unsafe { transmute(bytes) }
    }

    #[test]
    fn test_ccrnd_distinct_inputs_distinct_outputs() {
        let key = [0x2Bu8; 16];
        let public_s = bytes_to_block_safe([0xDEu8; 16]);
        let round_keys = unsafe { expand_aes128_key(&key) };
        let x1 = bytes_to_block_safe([0x01u8; 16]);
        let x2 = bytes_to_block_safe([0x02u8; 16]);
        let t = bytes_to_block_safe([0x05u8; 16]);
        let a: [u8; 16] = unsafe { transmute(ccrnd_with_round_keys(x1, t, &round_keys, public_s)) };
        let b: [u8; 16] = unsafe { transmute(ccrnd_with_round_keys(x2, t, &round_keys, public_s)) };
        assert_ne!(a, b);
    }

    #[test]
    fn test_ccrnd_distinct_tweaks_distinct_outputs() {
        let key = [0x2Bu8; 16];
        let public_s = bytes_to_block_safe([0xDEu8; 16]);
        let round_keys = unsafe { expand_aes128_key(&key) };
        let x = bytes_to_block_safe([0x42u8; 16]);
        let t1 = bytes_to_block_safe([0x05u8; 16]);
        let t2 = bytes_to_block_safe([0x06u8; 16]);
        let a: [u8; 16] = unsafe { transmute(ccrnd_with_round_keys(x, t1, &round_keys, public_s)) };
        let b: [u8; 16] = unsafe { transmute(ccrnd_with_round_keys(x, t2, &round_keys, public_s)) };
        assert_ne!(a, b);
    }
}
