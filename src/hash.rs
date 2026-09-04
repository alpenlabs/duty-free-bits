//! Label-aware CCRH: wraps the `Label`-free CCRH core ([`crate::crypto`]) with
//! the label↔block encoding the protocol's one-hot-scaling pads need.

use crate::crypto::{Block, expand};
use crate::label::Label;

/// `⌈log₂ modulus⌉` (number of bits needed to represent values < modulus).
///
/// Note: reducing a `⌈log₂ p⌉`-bit hash slice mod a non-power-of-two `p` —
/// which the CRT primes are — is non-uniform (residues `0..(2^⌈log₂ p⌉ − p)`
/// occur twice as often as the rest), so the IT-GC body does NOT sample its
/// `Z_p` pads this way: it rejection-samples wider windows to exact uniformity
/// (see `gc::body::pad_width`). This function only sizes the power-of-two
/// quantities — scaling-residue widths and `Z_{2^k}` ring words — where the
/// slice is exactly uniform.
pub fn lg_modulus(modulus: u64) -> usize {
    if modulus <= 1 {
        0
    } else {
        ((modulus - 1).ilog2() + 1) as usize
    }
}

/// Pack a boolean-label control into a 128-bit block for the CCRH's `x` input
/// (its bit-packed storage is exactly 2 u64 = 16 bytes — a direct copy).
fn label_to_block(l: &Label) -> Block {
    debug_assert_eq!(l.modulus(), 2, "control must be Z_2");
    let raw = l.raw_bits();
    debug_assert_eq!(raw.len(), 2, "Z_2 label must be exactly 2 u64 words");
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&raw[0].to_le_bytes());
    bytes[8..16].copy_from_slice(&raw[1].to_le_bytes());
    bytes
}

/// CCRH covering a whole group of one-hot scalings.
///
/// `total_bits` pseudorandom bits keyed on a boolean-label control, written
/// into the prefix of `out` (which must be at least `total_bits.div_ceil(8)`
/// long; bytes past the prefix are untouched, so callers may pack many groups
/// into one slab without per-group alloc). Bulk domain: bit 63 of the tweak
/// is set to disambiguate from the solo domain. `group_id` must be globally
/// fresh (paper App. A, Def. 4).
pub fn hash_bulk_into(ctrl_mask: &Label, group_id: usize, total_bits: usize, out: &mut [u8]) {
    let seed = label_to_block(ctrl_mask);
    debug_assert!(
        (group_id as u64) < (1u64 << 63),
        "group id uses the bulk-domain bit"
    );
    let domain = (group_id as u64) | (1u64 << 63);
    let len = total_bits.div_ceil(8);
    assert!(out.len() >= len, "hash_bulk_into: out shorter than output");
    expand(seed, domain, &mut out[..len]);
}

/// Continue a bulk CCRH stream from block `first_block`.
///
/// Fills all of `out` with blocks `first_block..` of the `(ctrl_mask,
/// group_id)` stream that [`hash_bulk_into`] starts at block 0 —
/// byte-identical to the corresponding suffix of one longer `hash_bulk_into`.
///
/// The per-block CTR counter, not the `group_id`, sequences blocks within a
/// stream (see [`crate::crypto::expand_from`]), so a caller that cannot bound
/// its consumption up front — the body's rejection sampler — extends the same
/// stream instead of reserving extra nonces. Def-4 freshness holds as long as
/// no block index is queried twice for a `(ctrl_mask, group_id)`.
pub fn hash_bulk_more(ctrl_mask: &Label, group_id: usize, first_block: u64, out: &mut [u8]) {
    let seed = label_to_block(ctrl_mask);
    debug_assert!(
        (group_id as u64) < (1u64 << 63),
        "group id uses the bulk-domain bit"
    );
    let domain = (group_id as u64) | (1u64 << 63);
    crate::crypto::expand_from(seed, domain, first_block, out);
}

/// CCRH for one one-hot scaling with a Z₂ payload, allocation-free.
///
/// 128 pseudorandom bits keyed on a packed boolean-label control, in the bulk
/// domain, returned as two little-endian words — for the steps whose working
/// representation is bare label words. `group_id` must be fresh per control
/// (see [`crate::crypto::nonce`]).
pub fn hash_z2(ctrl_words: &[u64; 2], group_id: u64) -> [u64; 2] {
    debug_assert!(group_id < (1u64 << 63), "group id uses the bulk-domain bit");
    let mut seed = [0u8; 16];
    seed[0..8].copy_from_slice(&ctrl_words[0].to_le_bytes());
    seed[8..16].copy_from_slice(&ctrl_words[1].to_le_bytes());
    let domain = group_id | (1u64 << 63);
    let mut out = [0u8; 16];
    expand(seed, domain, &mut out);
    [
        u64::from_le_bytes(out[0..8].try_into().unwrap()),
        u64::from_le_bytes(out[8..16].try_into().unwrap()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{LAMBDA, Label};
    use rand::RngExt;

    fn rand_ctrl() -> Label {
        let mut r = rand::rng();
        let coords: Vec<u64> = (0..LAMBDA).map(|_| r.random_range(0..2u64)).collect();
        Label::from_coords(&coords, 2)
    }

    /// Golden vector pinning the full CCRH pipeline (label→block, domain
    /// encoding, AES-CTR expansion) byte-for-byte for the bulk path.
    #[test]
    fn test_bulk_golden_vector() {
        let coords: Vec<u64> = (0..LAMBDA).map(|i| ((i * 7 + 3) % 5 == 0) as u64).collect();
        let ctrl = Label::from_coords(&coords, 2);
        let mut out = vec![0u8; 10];
        hash_bulk_into(&ctrl, 7, 80, &mut out);
        assert_eq!(out, [175, 72, 194, 184, 45, 188, 159, 234, 245, 163]);
    }

    /// `hash_bulk_more` must reproduce the tail of a longer `hash_bulk_into`
    /// exactly — the label-level mirror of `test_expand_from_continues_the_stream`.
    #[test]
    fn test_bulk_more_continues_bulk_into() {
        let s = rand_ctrl();
        let mut whole = vec![0u8; 5 * 16];
        hash_bulk_into(&s, 11, 5 * 128, &mut whole);
        for first_block in [1u64, 2, 4] {
            let start = first_block as usize * 16;
            let mut got = vec![0u8; whole.len() - start];
            hash_bulk_more(&s, 11, first_block, &mut got);
            assert_eq!(got, &whole[start..], "first_block={first_block}");
        }
    }

    #[test]
    fn test_hash_z2_deterministic_and_gid_sensitive() {
        let c = rand_ctrl();
        let words = [c.raw_bits()[0], c.raw_bits()[1]];
        assert_eq!(hash_z2(&words, 17), hash_z2(&words, 17));
        assert_ne!(hash_z2(&words, 1), hash_z2(&words, 2));
    }

    #[test]
    fn test_bulk_into_preserves_tail() {
        let s = rand_ctrl();
        let total_bits: usize = 5 * 9; // 6 bytes, not a multiple of 8
        let exact = total_bits.div_ceil(8);
        let mut out = vec![0xAAu8; exact + 8];
        hash_bulk_into(&s, 3, total_bits, &mut out);
        assert!(out[exact..].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_lg_modulus() {
        assert_eq!(lg_modulus(2), 1);
        assert_eq!(lg_modulus(3), 2);
        assert_eq!(lg_modulus(4), 2);
        assert_eq!(lg_modulus(409), 9);
        assert_eq!(lg_modulus(1), 0);
    }
}
