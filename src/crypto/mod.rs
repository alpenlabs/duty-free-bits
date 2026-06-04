//! Correlation-robust hash (CCRH) core — the bridge primitive between the
//! computational GC (CF labels) and the information-theoretic GC (NCF shares).
//!
//! Instantiated using the CCRND construction from §5 of
//! <https://eprint.iacr.org/2019/074.pdf>:
//!
//!   H(x, t) = AES_K( σ(x ⊕ public_s ⊕ t) ) ⊕ σ(x ⊕ public_s ⊕ t)
//!
//! with a fixed AES key (a public permutation) and fixed `public_s`. σ is the
//! linear orthomorphism `(L||R) → (L⊕R)||L`. Per paper Def. 4, a sequence of
//! `H(x, nonce)` queries is *legal* iff no `(x, nonce)` pair repeats; reusing one
//! reuses the one-time pad it produces. Callers are responsible for fresh nonces
//! (see [`nonce`]).
//!
//! This module is **`Label`-free**: it operates on `[u8; 16]` blocks only. The
//! `Label`↔block facade (`hash_solo` / `hash_bulk` / `extract_ncf`) lives in
//! [`crate::garble::hash`]. The AES backend is target-selected: aarch64 uses
//! NEON/AES-NI; other targets use a portable software AES that is byte-identical.

/// aarch64 NEON/AES-NI backend.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
/// Portable software-AES backend (byte-identical to NEON).
pub mod portable;
pub mod nonce;

use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
use aarch64 as backend;
#[cfg(not(target_arch = "aarch64"))]
use portable as backend;
use backend::RoundKeys;

/// A 128-bit CCRH block.
pub type Block = [u8; 16];

/// Fixed AES-128 key (public permutation key for the CCRH). Bytes from the
/// fractional part of the golden ratio — a "nothing-up-my-sleeve" constant.
const CCRH_KEY: [u8; 16] = [
    0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15,
    0xf3, 0x9c, 0xc0, 0x60, 0x5c, 0xed, 0xc8, 0x34,
];

/// Fixed `public_s` (CCRND's tweakable-PRF public string).
const CCRH_PUBLIC_S: [u8; 16] = [
    0xa0, 0x9e, 0x66, 0x7f, 0x3b, 0xcc, 0x90, 0x8b,
    0xb6, 0x7a, 0xe8, 0x58, 0x4c, 0xaa, 0x73, 0xb2,
];

/// Process-global pre-expanded round keys (computed once on first use).
fn round_keys() -> &'static RoundKeys {
    static C: OnceLock<RoundKeys> = OnceLock::new();
    C.get_or_init(|| backend::expand_key(&CCRH_KEY))
}

/// AES-CTR expansion under CCRND: fill `output` from `(seed, nonce)`.
///
/// Each 16-byte block uses a tweak with `nonce` in the low 64 bits and a per-block
/// counter in the high 64 bits, so blocks never collide.
pub fn expand(seed: Block, nonce: u64, output: &mut [u8]) {
    let keys = round_keys();
    let mut counter: u64 = 0;
    let mut written = 0;
    while written < output.len() {
        let tweak = ((nonce as u128) | ((counter as u128) << 64)).to_le_bytes();
        let h = backend::ccrnd(seed, tweak, keys, CCRH_PUBLIC_S);
        let take = (output.len() - written).min(16);
        output[written..written + take].copy_from_slice(&h[..take]);
        written += take;
        counter += 1;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    /// The portable software backend must be byte-identical to the NEON/AES-NI
    /// backend — otherwise the off-aarch64 build runs *different* crypto.
    #[test]
    fn test_portable_backend_matches_neon() {
        use super::{CCRH_KEY, CCRH_PUBLIC_S, aarch64, portable};
        let rk_neon = aarch64::expand_key(&CCRH_KEY);
        let rk_soft = portable::expand_key(&CCRH_KEY);
        for i in 0..128u64 {
            let mut seed = [0u8; 16];
            seed[0..8].copy_from_slice(&i.to_le_bytes());
            seed[8] = 0xa5;
            seed[15] = (i as u8).wrapping_mul(31);
            let mut tweak = [0u8; 16];
            tweak[0..8].copy_from_slice(&(i.wrapping_mul(7).wrapping_add(1)).to_le_bytes());
            tweak[8..16].copy_from_slice(&((1u64 << 63) | i).to_le_bytes());
            assert_eq!(
                aarch64::ccrnd(seed, tweak, &rk_neon, CCRH_PUBLIC_S),
                portable::ccrnd(seed, tweak, &rk_soft, CCRH_PUBLIC_S),
                "portable vs NEON mismatch at i={i}",
            );
        }
    }
}
