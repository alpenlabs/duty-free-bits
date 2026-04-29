//! CCRH stub used by switch gates.
//!
//! This is a **non-cryptographic** placeholder — quite literally the zero
//! function. The point-and-permute LSB the garbler emits still recovers
//! `ctrl`, and the evaluator's backward-propagation through joins and adds
//! fills in labels for switches that don't fire, so correctness does not
//! depend on `H` being non-trivial. This exists only to be swapped for a real
//! CCRH (e.g. AES-MMO) later.

use super::label::Label;
use crate::types::GateId;

/// `H(ctrl_mask, gid) = 0` for now.
pub fn hash(ctrl_mask: &Label, gid: GateId, out_is_cf: bool, out_modulus: u64) -> Label {
    debug_assert!(
        matches!(ctrl_mask, Label::Cf(c) if c.modulus() == 2),
        "ctrl mask must be CF Z_2"
    );
    let _ = gid;
    Label::zero(out_is_cf, out_modulus)
}

#[cfg(test)]
mod tests {
    use super::super::label::{CfLabel, LAMBDA, Label};
    use super::*;

    fn rand_ctrl() -> Label {
        use rand::Rng;
        let mut r = rand::rng();
        let coords: Vec<u64> = (0..LAMBDA).map(|_| r.random_range(0..2u64)).collect();
        Label::Cf(CfLabel::from_coords(&coords, 2))
    }

    #[test]
    fn test_outputs_zero_cf() {
        let s = rand_ctrl();
        let h = hash(&s, 42, true, 1 << 10);
        assert_eq!(h, Label::zero_cf(1 << 10));
    }

    #[test]
    fn test_outputs_zero_ncf() {
        let s = rand_ctrl();
        let h = hash(&s, 42, false, 409);
        assert_eq!(h, Label::zero_ncf(409));
    }
}
