//! aarch64-specific intrinsics: AES-NI key schedule + CCRND hash.
//!
//! Lifted from the `gobble` crate (`/Users/nakul/ckt/crates/gobble/src/aarch64/mod.rs`).
//! That crate uses these primitives for privacy-free half-gates; we use them
//! for our switch-system CCRH (`hash::hash_solo` / `hash::hash_bulk`). The
//! protocol is different but the underlying H requirement is the same:
//! one tweakable correlation-robust hash from a 128-bit input + 128-bit
//! tweak to a 128-bit output.
//!
//! `ccrnd_with_round_keys` implements the CCRND construction from §5 of
//! <https://eprint.iacr.org/2019/074.pdf>:
//!
//!   H(x, t) = AES_K( σ(x ⊕ public_s ⊕ t) ) ⊕ σ(x ⊕ public_s ⊕ t)
//!
//! where σ is the linear orthomorphism `(L||R) → (L⊕R)||L`. One AES call
//! plus a few XORs and a NEON `vextq` shuffle.

use std::arch::aarch64::*;
use std::mem::transmute;

mod expand;

use expand::expand_key;

/// Pre-expanded AES-128 round keys (11 round keys for the standard schedule).
pub type Aes128RoundKeys = expand::Aes128RoundKeys;

/// AES-128 key expansion.
///
/// # Safety
/// CPU must support `aes` target feature.
#[target_feature(enable = "aes")]
pub unsafe fn expand_aes128_key(key: &[u8; 16]) -> Aes128RoundKeys {
    unsafe { expand_key::<16, 11>(key) }
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

/// Convert little-endian bytes to a 128-bit register.
///
/// # Safety
/// CPU must support `neon` (the result is a `uint8x16_t`).
#[inline]
pub unsafe fn bytes_to_block(bytes: [u8; 16]) -> uint8x16_t {
    unsafe { transmute(bytes) }
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
        let a: [u8; 16] =
            unsafe { transmute(ccrnd_with_round_keys(x1, t, &round_keys, public_s)) };
        let b: [u8; 16] =
            unsafe { transmute(ccrnd_with_round_keys(x2, t, &round_keys, public_s)) };
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
        let a: [u8; 16] =
            unsafe { transmute(ccrnd_with_round_keys(x, t1, &round_keys, public_s)) };
        let b: [u8; 16] =
            unsafe { transmute(ccrnd_with_round_keys(x, t2, &round_keys, public_s)) };
        assert_ne!(a, b);
    }
}
