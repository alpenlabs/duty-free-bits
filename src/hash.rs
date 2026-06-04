//! Label-aware CCRH facade for switch gates.
//!
//! Wraps the `Label`-free CCRH core ([`crate::crypto`]) with the label↔block
//! encoding and the per-output sizing the garbler/evaluator need:
//!
//!   * [`hash_solo`] — one switch's H output, sized for its output wire.
//!   * [`hash_bulk`] — one wide H output for a switch group; sliced via
//!     [`extract_ncf`] across NCF members.
//!
//! Switch controls are always CF Z_2, so the hash seed is exactly 128 bits.
//! The `nonce` passed to the core is the gate id (solo) or the group id with the
//! bulk flag set; the caller guarantees freshness (see [`crate::crypto::nonce`]).

use crate::crypto::{Block, expand};
use crate::label::{self, CfLabel, LAMBDA, Label, NcfLabel};
use crate::types::GateId;

/// `⌈log₂ modulus⌉` (number of bits needed to represent values < modulus).
pub fn lg_modulus(modulus: u64) -> usize {
    if modulus <= 1 {
        0
    } else {
        ((modulus - 1).ilog2() + 1) as usize
    }
}

/// Compress a label down to a 128-bit block suitable as CCRND's `x` input.
///
/// In our protocol switch ctrls are always CF Z_2, so the input is exactly
/// 128 bits and we just copy the bit-packed storage. (Wider inputs are folded
/// by XOR'ing 16-byte chunks; in practice we never call H on those.)
fn label_to_block(l: &Label) -> Block {
    match l {
        Label::Cf(c) => {
            let raw = c.raw_bits();
            // Z_2 case: exactly 2 u64 = 16 bytes.
            if raw.len() == 2 {
                let mut bytes = [0u8; 16];
                bytes[0..8].copy_from_slice(&raw[0].to_le_bytes());
                bytes[8..16].copy_from_slice(&raw[1].to_le_bytes());
                bytes
            } else {
                // Wider CF: XOR-fold the raw words into one 16-byte block.
                // Not used by switches today, but defined to be deterministic.
                let mut acc = [0u8; 16];
                for (i, &w) in raw.iter().enumerate() {
                    let lane = (i & 1) * 8;
                    let bytes = w.to_le_bytes();
                    for b in 0..8 {
                        acc[lane + b] ^= bytes[b];
                    }
                }
                acc
            }
        }
        Label::Ncf(n) => {
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&n.rep.to_le_bytes());
            bytes[8..16].copy_from_slice(&n.modulus.to_le_bytes());
            bytes
        }
    }
}

/// CCRH for a single switch gate.
pub fn hash_solo(ctrl_mask: &Label, gid: GateId, out_is_cf: bool, out_modulus: u64) -> Label {
    debug_assert!(
        matches!(ctrl_mask, Label::Cf(c) if c.modulus() == 2),
        "ctrl mask must be CF Z_2"
    );
    let seed = label_to_block(ctrl_mask);
    // Bit 63 is the solo/bulk domain flag, so the nonce must leave it clear.
    debug_assert!((gid as u64) < (1u64 << 63), "switch gid uses the bulk-domain bit");
    let domain = gid as u64; // solo: bit 63 = 0.
    if out_is_cf {
        let k = out_modulus.trailing_zeros() as usize;
        let total_bits = LAMBDA * k;
        let words = total_bits.div_ceil(64);
        let mut buf = vec![0u8; words * 8];
        expand(seed, domain, &mut buf);
        let bits: Vec<u64> = (0..words)
            .map(|w| u64::from_le_bytes(buf[w * 8..(w + 1) * 8].try_into().unwrap()))
            .collect();
        Label::Cf(CfLabel::from_raw_bits(bits, out_modulus))
    } else {
        let lg_m = lg_modulus(out_modulus);
        let mut buf = vec![0u8; lg_m.div_ceil(8)];
        expand(seed, domain, &mut buf);
        let mut value = 0u64;
        for b in 0..lg_m {
            if (buf[b / 8] >> (b % 8)) & 1 == 1 {
                value |= 1u64 << b;
            }
        }
        Label::Ncf(NcfLabel {
            rep: if out_modulus == 0 { 0 } else { value % out_modulus },
            modulus: out_modulus,
        })
    }
}

/// CCRH for a switch group: one wide call covering all members.
pub fn hash_bulk(ctrl_mask: &Label, group_id: usize, total_bits: usize) -> Vec<u8> {
    debug_assert!(
        matches!(ctrl_mask, Label::Cf(c) if c.modulus() == 2),
        "ctrl mask must be CF Z_2"
    );
    let seed = label_to_block(ctrl_mask);
    // Bulk: set bit 63 of the domain to disambiguate from solo.
    // Below the solo/bulk flag bit, so distinct group ids stay in the bulk domain.
    debug_assert!((group_id as u64) < (1u64 << 63), "group id uses the bulk-domain bit");
    let domain = (group_id as u64) | (1u64 << 63);
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    expand(seed, domain, &mut out);
    out
}

/// Extract member `idx`'s NCF label from a wide bulk-hash output.
pub fn extract_ncf(wide: &[u8], idx: usize, modulus: u64) -> Label {
    let lg_m = lg_modulus(modulus);
    debug_assert!(lg_m <= 64, "extract_ncf: modulus {} too large for u64", modulus);
    let bit_off = idx * lg_m;
    let mut acc: u64 = 0;
    for i in 0..lg_m {
        let bit = bit_off + i;
        debug_assert!(bit / 8 < wide.len(), "extract_ncf: slice overrun");
        let b = (wide[bit / 8] >> (bit % 8)) & 1;
        acc |= (b as u64) << i;
    }
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
