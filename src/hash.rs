//! Label-aware CCRH for switch gates.
//!
//! Wraps the `Label`-free CCRH core ([`crate::crypto`]) with the label↔block encoding.

use crate::crypto::{Block, expand};
use crate::label::{self, CfLabel, LAMBDA, Label, NcfLabel};

/// `⌈log₂ modulus⌉` (number of bits needed to represent values < modulus).
///
/// TODO: For a non-power-of-two NCF modulus `p`, this
/// tight width makes an extracted slice's `value % p` *non-uniform* — residues
/// `0..(2^⌈log₂ p⌉ − p)` occur twice, the rest once, which might leak something.
pub fn lg_modulus(modulus: u64) -> usize {
    if modulus <= 1 {
        0
    } else {
        ((modulus - 1).ilog2() + 1) as usize
    }
}

/// Convert a switch control label to a 128-bit block for CCRND's `x` input.
///
/// Switch controls are always CF Z_2, whose bit-packed storage is exactly
/// 2 u64 = 16 bytes — so this is a direct copy. Any other label kind is a
/// caller bug.
fn label_to_block(l: &Label) -> Block {
    let Label::Cf(c) = l else {
        panic!("label_to_block: switch control must be CF Z_2, got NCF");
    };
    debug_assert_eq!(c.modulus(), 2, "label_to_block: switch control must be Z_2");
    let raw = c.raw_bits();
    debug_assert_eq!(raw.len(), 2, "CF Z_2 label must be exactly 2 u64 words");
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&raw[0].to_le_bytes());
    bytes[8..16].copy_from_slice(&raw[1].to_le_bytes());
    bytes
}

/// CCRH for a single switch gate.
///
/// `nonce` must be globally fresh for the garbling (paper App. A, Def. 4: no
/// two queries share a nonce) — callers offset per-phase gate ids by a
/// monotonically allocated base.
pub fn hash_solo(ctrl_mask: &Label, nonce: u64, out_is_cf: bool, out_modulus: u64) -> Label {
    debug_assert!(
        matches!(ctrl_mask, Label::Cf(c) if c.modulus() == 2),
        "ctrl mask must be CF Z_2"
    );
    let seed = label_to_block(ctrl_mask);
    // Bit 63 is the solo/bulk domain flag, so the nonce must leave it clear.
    debug_assert!(nonce < (1u64 << 63), "solo nonce uses the bulk-domain bit");
    let domain = nonce; // solo: bit 63 = 0.
    if out_is_cf {
        let k = out_modulus.trailing_zeros() as usize;
        if k == 1 {
            // Z_2: exactly LAMBDA = 128 bits = 16 bytes, one AES block.
            let mut buf = [0u8; 16];
            expand(seed, domain, &mut buf);
            return Label::Cf(CfLabel::from_packed_bytes(&buf, out_modulus));
        }
        let words = (LAMBDA * k).div_ceil(64);
        // k ≤ 32 ⇒ words ≤ ⌈128·32/64⌉ = 64 ⇒ 512 bytes bound the output.
        debug_assert!(words * 8 <= 512, "CF modulus 2^{k} exceeds the u32 lanes");
        let mut buf = [0u8; 512];
        let buf = &mut buf[..words * 8];
        expand(seed, domain, buf);
        Label::Cf(CfLabel::from_packed_bytes(buf, out_modulus))
    } else {
        let lg_m = lg_modulus(out_modulus);
        debug_assert!(lg_m <= 64, "NCF modulus {} too large for u64", out_modulus);
        let mut buf = [0u8; 8];
        expand(seed, domain, &mut buf[..lg_m.div_ceil(8)]);
        // Unfilled high bytes are zero, so the LE word load masked to lg_m
        // bits equals the LSB-first per-bit assembly of the expanded bytes.
        let word = u64::from_le_bytes(buf);
        let value = if lg_m >= 64 {
            word
        } else {
            word & ((1u64 << lg_m) - 1)
        };
        // value < 2^lg_m ≤ 2·(out_modulus − 1) < 2·out_modulus (lg_m = ⌈log₂ m⌉),
        // so one compare-subtract equals `value % out_modulus`. For modulus ≤ 1,
        // lg_m = 0 forces value = 0, consistent with the `out_modulus == 0` guard.
        let rep = if out_modulus != 0 && value >= out_modulus {
            value - out_modulus
        } else {
            value
        };
        Label::Ncf(NcfLabel {
            rep,
            modulus: out_modulus,
        })
    }
}

/// CCRH for a switch group: one wide call covering all members.
pub fn hash_bulk(ctrl_mask: &Label, group_id: usize, total_bits: usize) -> Vec<u8> {
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    hash_bulk_into(ctrl_mask, group_id, total_bits, &mut out);
    out
}

/// As [`hash_bulk`], but writes the `total_bits.div_ceil(8)` output bytes into
/// the prefix of `out` (which must be at least that long); bytes past the
/// prefix are left untouched.
///
/// Byte-identical to [`hash_bulk`] — callers may pack many groups into one
/// slab without per-group allocation.
pub fn hash_bulk_into(ctrl_mask: &Label, group_id: usize, total_bits: usize, out: &mut [u8]) {
    debug_assert!(
        matches!(ctrl_mask, Label::Cf(c) if c.modulus() == 2),
        "ctrl mask must be CF Z_2"
    );
    let seed = label_to_block(ctrl_mask);
    // Bulk: set bit 63 of the domain to disambiguate from solo.
    // Below the solo/bulk flag bit, so distinct group ids stay in the bulk domain.
    debug_assert!(
        (group_id as u64) < (1u64 << 63),
        "group id uses the bulk-domain bit"
    );
    let domain = (group_id as u64) | (1u64 << 63);
    let len = total_bits.div_ceil(8);
    assert!(out.len() >= len, "hash_bulk_into: out shorter than output");
    expand(seed, domain, &mut out[..len]);
}

/// CCRH for one Z₂-payload kernel switch, allocation-free.
///
/// 128 pseudorandom bits keyed on a packed CF Z₂ control label, in the bulk
/// domain — byte-identical to [`hash_bulk`] on the same control and
/// `group_id` with `total_bits = 128`, reinterpreted as two little-endian
/// words. For kernels whose working representation is bare label words.
/// `group_id` must be fresh per control (see [`crate::crypto::nonce`]).
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

/// Extract member `idx`'s NCF label from a wide bulk-hash output.
pub fn extract_ncf(wide: &[u8], idx: usize, modulus: u64) -> Label {
    let lg_m = lg_modulus(modulus);
    debug_assert!(
        lg_m <= 64,
        "extract_ncf: modulus {} too large for u64",
        modulus
    );
    let bit_off = idx * lg_m;
    let mut acc: u64 = 0;
    for i in 0..lg_m {
        let bit = bit_off + i;
        debug_assert!(bit / 8 < wide.len(), "extract_ncf: slice overrun");
        let b = (wide[bit / 8] >> (bit % 8)) & 1;
        acc |= (b as u64) << i;
    }
    // TODO: `acc` is exactly `lg_m = ⌈log₂ modulus⌉` bits, so `acc % modulus` is slightly biased toward small residues for a non-power-of-two `modulus`.
    let rep = if modulus == 0 { 0 } else { acc % modulus };
    Label::Ncf(NcfLabel { rep, modulus })
}

/// Re-export so external callers can build matching Δ_R material.
pub fn delta_r(delta: u128, modulus: u64) -> CfLabel {
    label::delta_r(delta, modulus)
}

#[cfg(test)]
mod tests {
    use super::super::label::{CfLabel, Label};
    use super::*;

    fn rand_ctrl() -> Label {
        use rand::Rng;
        let mut r = rand::rng();
        let coords: Vec<u64> = (0..LAMBDA).map(|_| r.random_range(0..2u64)).collect();
        Label::Cf(CfLabel::from_coords(&coords, 2))
    }

    #[test]
    fn test_solo_deterministic() {
        let s = rand_ctrl();
        let a = hash_solo(&s, 17, true, 1 << 10);
        let b = hash_solo(&s, 17, true, 1 << 10);
        assert_eq!(a, b);
    }

    #[test]
    fn test_solo_gid_changes_output() {
        let s = rand_ctrl();
        let a = hash_solo(&s, 1, true, 1 << 10);
        let b = hash_solo(&s, 2, true, 1 << 10);
        assert_ne!(a, b);
    }

    /// Golden vectors pinning the full CCRH pipeline (label→block, domain
    /// encoding, AES-CTR expansion, NCF bit extraction) byte-for-byte. R1
    /// refactors hash.rs behind a `Ccrh` trait; these must not change.
    #[test]
    fn test_ccrh_golden_vectors() {
        let coords: Vec<u64> = (0..LAMBDA).map(|i| ((i * 7 + 3) % 5 == 0) as u64).collect();
        let ctrl = Label::Cf(CfLabel::from_coords(&coords, 2));

        match hash_solo(&ctrl, 42, false, 31) {
            Label::Ncf(n) => assert_eq!((n.rep, n.modulus), (20, 31), "hash_solo NCF golden"),
            _ => panic!("expected NCF"),
        }
        assert_eq!(
            hash_bulk(&ctrl, 7, 80),
            [175, 72, 194, 184, 45, 188, 159, 234, 245, 163],
            "hash_bulk golden"
        );
    }

    #[test]
    fn test_solo_ctrl_changes_output() {
        let mut r = rand::rng();
        use rand::Rng;
        let s_a: Vec<u64> = (0..LAMBDA).map(|_| r.random_range(0..2u64)).collect();
        let mut s_b = s_a.clone();
        s_b[0] ^= 1;
        let la = Label::Cf(CfLabel::from_coords(&s_a, 2));
        let lb = Label::Cf(CfLabel::from_coords(&s_b, 2));
        let a = hash_solo(&la, 5, false, 409);
        let b = hash_solo(&lb, 5, false, 409);
        assert_ne!(a, b);
    }

    #[test]
    fn test_bulk_deterministic_and_correct_length() {
        let s = rand_ctrl();
        let total_bits = 5 * 9;
        let a = hash_bulk(&s, 0, total_bits);
        let b = hash_bulk(&s, 0, total_bits);
        assert_eq!(a, b);
        assert_eq!(a.len(), total_bits.div_ceil(8));
    }

    #[test]
    fn test_bulk_into_matches_bulk_and_preserves_tail() {
        let s = rand_ctrl();
        let total_bits: usize = 5 * 9; // exact length 6 bytes, not a multiple of 8.
        let exact_len = total_bits.div_ceil(8);
        let expected = hash_bulk(&s, 3, total_bits);
        let mut out = vec![0xAAu8; exact_len + 8];
        hash_bulk_into(&s, 3, total_bits, &mut out);
        assert_eq!(&out[..exact_len], &expected[..]);
        assert!(out[exact_len..].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_bulk_group_id_changes_output() {
        let s = rand_ctrl();
        let a = hash_bulk(&s, 0, 64);
        let b = hash_bulk(&s, 1, 64);
        assert_ne!(a, b);
    }

    #[test]
    fn test_solo_bulk_no_collision() {
        // Solo with gid g and bulk with group_id g should not collide
        // (the high bit in `domain` distinguishes them).
        let s = rand_ctrl();
        let solo = hash_solo(&s, 7, false, 256);
        let bulk = hash_bulk(&s, 7, 8);
        let solo_rep = match solo {
            Label::Ncf(n) => n.rep,
            _ => panic!(),
        };
        let bulk_rep = bulk[0] as u64;
        assert_ne!(solo_rep, bulk_rep);
    }

    #[test]
    fn test_extract_ncf_recovers_planted_bits() {
        let lg_m: usize = 4;
        let n: usize = 16;
        let mut wide = vec![0u8; (n * lg_m).div_ceil(8)];
        for i in 0..n {
            let bit_off = i * lg_m;
            for b in 0..lg_m {
                let bit = bit_off + b;
                let v = ((i as u64) >> b) & 1;
                wide[bit / 8] |= (v as u8) << (bit % 8);
            }
        }
        for i in 0..n {
            let l = extract_ncf(&wide, i, 16);
            match l {
                Label::Ncf(n) => assert_eq!(n.rep, i as u64),
                _ => panic!("expected NCF"),
            }
        }
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
