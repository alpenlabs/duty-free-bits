//! The crate's concrete CCRH (circular correlation-robust hash) — the CCRND
//! construction from §5 of <https://eprint.iacr.org/2019/074.pdf>.
//!
//! "CCRND" and "CCRH" name the same primitive here; downstream modules say CCRH.
//!
//!   H(x, t) = AES_K( σ(x ⊕ public_s ⊕ t) ) ⊕ σ(x ⊕ public_s ⊕ t)
//!
//! with a fixed AES key (a public permutation) and fixed `public_s`. σ is the
//! linear orthomorphism `(L||R) → (L⊕R)||L`. Per paper Def. 4, a sequence of
//! `H(x, nonce)` queries is *legal* iff no `(x, nonce)` pair repeats; reusing one
//! reuses the one-time pad it produces. Callers are responsible for fresh nonces.
//!
//! This module operates on `[u8; 16]` blocks only. The AES backend is target-selected: aarch64 uses
//! NEON/AES-NI; other targets use a portable software AES that is byte-identical.

/// aarch64 NEON/AES-NI backend.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod nonce;
/// Portable software-AES backend (byte-identical to NEON).
pub mod portable;

use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
use aarch64 as backend;
use backend::RoundKeys;
#[cfg(not(target_arch = "aarch64"))]
use portable as backend;

/// A 128-bit CCRH block.
pub type Block = [u8; 16];

/// Process-global CCRH block counter (only compiled under `count-hashes`).
#[cfg(feature = "count-hashes")]
static HASH_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// CCRH blocks (16-byte AES outputs) emitted so far.
///
/// Returns 0 unless built with the `count-hashes` feature — used for per-party
/// hash accounting in benchmarks (snapshot the delta around a garble or eval region).
#[inline]
pub fn hash_blocks() -> u64 {
    #[cfg(feature = "count-hashes")]
    {
        HASH_BLOCKS.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "count-hashes"))]
    {
        0
    }
}

/// Reset the CCRH block counter (no-op without `count-hashes`).
#[inline]
pub fn reset_hash_blocks() {
    #[cfg(feature = "count-hashes")]
    HASH_BLOCKS.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Fixed AES-128 key (public permutation key for the CCRH). Bytes from the
/// fractional part of the golden ratio — a "nothing-up-my-sleeve" constant.
const CCRH_KEY: [u8; 16] = [
    0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60, 0x5c, 0xed, 0xc8, 0x34,
];

/// Fixed `public_s` (CCRND's tweakable-PRF public string).
const CCRH_PUBLIC_S: [u8; 16] = [
    0xa0, 0x9e, 0x66, 0x7f, 0x3b, 0xcc, 0x90, 0x8b, 0xb6, 0x7a, 0xe8, 0x58, 0x4c, 0xaa, 0x73, 0xb2,
];

/// Process-global pre-expanded round keys (computed once on first use).
fn round_keys() -> &'static RoundKeys {
    static C: OnceLock<RoundKeys> = OnceLock::new();
    C.get_or_init(|| backend::expand_key(&CCRH_KEY))
}

/// Build the CTR tweak for block `counter`: `nonce` low 64 bits, counter high.
#[inline]
fn ctr_tweak(nonce: u64, counter: u64) -> Block {
    ((nonce as u128) | ((counter as u128) << 64)).to_le_bytes()
}

/// AES-CTR expansion under CCRND: fill `output` from `(seed, nonce)`.
///
/// Each 16-byte block uses a tweak with `nonce` in the low 64 bits and a per-block
/// counter in the high 64 bits, so blocks never collide.
pub fn expand(seed: Block, nonce: u64, output: &mut [u8]) {
    let keys = round_keys();
    let mut counter: u64 = 0;
    let mut written = 0;

    // Grouped path (aarch64): `INTERLEAVE` independent CTR blocks in flight
    // through the round-major kernel. Same tweak sequence as the tail loop,
    // and each block is byte-identical to `backend::ccrnd`, so output bytes
    // match the one-block-at-a-time loop exactly.
    #[cfg(target_arch = "aarch64")]
    {
        const W: usize = backend::INTERLEAVE;
        let blocks =
            backend::ccrnd_ctr_fill::<W>(seed, nonce, counter, keys, CCRH_PUBLIC_S, output);
        counter += blocks as u64;
        written += blocks * 16;
    }

    // Tail (and the whole output on portable targets): one block at a time.
    while written < output.len() {
        let h = backend::ccrnd(seed, ctr_tweak(nonce, counter), keys, CCRH_PUBLIC_S);
        let take = (output.len() - written).min(16);
        output[written..written + take].copy_from_slice(&h[..take]);
        written += take;
        counter += 1;
    }

    #[cfg(feature = "count-hashes")]
    HASH_BLOCKS.fetch_add(
        output.len().div_ceil(16) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[cfg(test)]
mod expand_tests {
    use super::*;

    /// The pre-interleaving `expand` loop (one CCRND block at a time), inlined
    /// here verbatim as the differential oracle for the grouped path.
    fn expand_oracle(seed: Block, nonce: u64, output: &mut [u8]) {
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

    #[test]
    fn test_expand_matches_single_block_oracle() {
        let seeds: [Block; 4] = [
            [0u8; 16],
            [0xff; 16],
            std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11)),
            std::array::from_fn(|i| 0xa5u8.rotate_left(i as u32)),
        ];
        let nonces = [0u64, 1, 0xdead_beef_cafe_f00d, u64::MAX];
        for &seed in &seeds {
            for &nonce in &nonces {
                for len in [1usize, 8, 15, 16, 17, 64, 144, 145, 352, 512] {
                    // Distinct fill bytes so untouched bytes can't match by luck.
                    let mut got = vec![0xAAu8; len];
                    let mut want = vec![0x55u8; len];
                    expand(seed, nonce, &mut got);
                    expand_oracle(seed, nonce, &mut want);
                    assert_eq!(
                        got, want,
                        "expand mismatch: seed={seed:02x?} nonce={nonce:#x} len={len}",
                    );
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    /// Indicative ns/block for the single vs interleaved CCRND paths.
    /// Run: `cargo test --release bench_ccrnd -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_ccrnd_single_vs_interleaved() {
        use super::{Block, CCRH_PUBLIC_S, backend, expand, round_keys};
        use backend::RoundKeys;
        use std::hint::black_box;
        use std::time::Instant;

        const BLOCKS: usize = 1 << 20; // ~1M blocks per timed leg.
        // 4096-block (64 KiB) buffer per fill: keeps the stores in L1 so we
        // time the AES pipes, not memory bandwidth.
        const BUF_BLOCKS: usize = 4096;

        // The kernel alone, one block per call (still overlapped across loop
        // iterations by the out-of-order core — blocks are independent).
        fn bench_single_kernel(keys: &RoundKeys) -> f64 {
            let seed = black_box([0x42u8; 16]);
            let mut sink = 0u8;
            let t = Instant::now();
            for c in 0..BLOCKS as u64 {
                let tweak = (7u128 | ((c as u128) << 64)).to_le_bytes();
                sink ^= backend::ccrnd(seed, tweak, keys, CCRH_PUBLIC_S)[0];
            }
            black_box(sink);
            t.elapsed().as_nanos() as f64 / BLOCKS as f64
        }

        // The interleaved kernel at width `W`.
        fn bench_fill<const W: usize>(keys: &RoundKeys) -> f64 {
            let mut buf = vec![0u8; BUF_BLOCKS * 16];
            let seed = black_box([0x42u8; 16]);
            let mut blocks = 0usize;
            let t = Instant::now();
            for i in 0..BLOCKS / BUF_BLOCKS {
                blocks += backend::ccrnd_ctr_fill::<W>(
                    seed,
                    7,
                    (i * BUF_BLOCKS) as u64,
                    keys,
                    CCRH_PUBLIC_S,
                    &mut buf,
                );
                black_box(&buf);
            }
            t.elapsed().as_nanos() as f64 / blocks as f64
        }

        // The full `expand` entry point, old (one-block loop, inlined oracle
        // from `expand_tests`) vs new (grouped path).
        fn bench_expand(old: bool, keys: &RoundKeys) -> f64 {
            let mut buf = vec![0u8; BUF_BLOCKS * 16];
            let seed = black_box([0x42u8; 16]);
            let t = Instant::now();
            for i in 0..BLOCKS / BUF_BLOCKS {
                if old {
                    let mut counter: u64 = 0;
                    let mut written = 0;
                    while written < buf.len() {
                        let tweak = ((i as u128) | ((counter as u128) << 64)).to_le_bytes();
                        let h: Block = backend::ccrnd(seed, tweak, keys, CCRH_PUBLIC_S);
                        let take = (buf.len() - written).min(16);
                        buf[written..written + take].copy_from_slice(&h[..take]);
                        written += take;
                        counter += 1;
                    }
                } else {
                    expand(seed, i as u64, &mut buf);
                }
                black_box(&buf);
            }
            t.elapsed().as_nanos() as f64 / BLOCKS as f64
        }

        let keys = round_keys();
        // Warmup + measure.
        for _ in 0..2 {
            let single = bench_single_kernel(keys);
            let wide4 = bench_fill::<4>(keys);
            let wide8 = bench_fill::<8>(keys);
            let old = bench_expand(true, keys);
            let new = bench_expand(false, keys);
            eprintln!(
                "kernel single: {single:>6.3} ns/block | kernel x4: {wide4:>6.3} | kernel x8: \
                 {wide8:>6.3} | expand old: {old:>6.3} | expand new: {new:>6.3}",
            );
        }
    }

    /// The portable software backend must be byte-identical to the NEON/AES-NI
    /// backend — otherwise the off-aarch64 build runs different crypto.
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
