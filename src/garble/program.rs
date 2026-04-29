//! Garbled program encoding.
//!
//! The program carries:
//! * one **switch LSB** per `Switch` gate — the low bit of its control-wire mask.
//!   With point-and-permute (`Δ` forced to have low bit 1), XORing this with the
//!   evaluator's ctrl label lsb recovers the ctrl value in the clear.
//! * one **join diff** (`X_in0 − X_in1`) per `Join` gate, letting the evaluator
//!   translate between the two sides' masks.
//! * one **output mask** per declared NCF output wire.
//!
//! `total_bits()` equals the paper's `join_width(S) + |Gout|` plus a negligible
//! 1-bit-per-switch overhead for ctrl LSBs.

use super::label::Label;
use crate::types::GateId;

/// Per-gate material.
#[derive(Clone, Debug)]
pub enum GateMaterial {
    /// Low bit of the control-wire mask for a switch (point-and-permute).
    SwitchLsb(bool),
    /// `X_in0 − X_in1` for a join gate.
    JoinDiff(Label),
}

/// Garbled program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Material indexed by gate id. `None` for gates that emit nothing.
    per_gate: Vec<Option<GateMaterial>>,
    /// One mask per declared output wire, in the caller's order.
    output_masks: Vec<Label>,
    /// Accumulated bit count.
    bits: usize,
}

impl Program {
    /// New program sized for `num_gates`.
    pub fn with_num_gates(num_gates: usize) -> Self {
        Self {
            per_gate: vec![None; num_gates],
            output_masks: Vec::new(),
            bits: 0,
        }
    }

    /// Record a switch's control LSB.
    pub fn set_switch_lsb(&mut self, gid: GateId, lsb: bool) {
        assert!(
            self.per_gate[gid].is_none(),
            "per-gate material already set for gate {}",
            gid
        );
        self.per_gate[gid] = Some(GateMaterial::SwitchLsb(lsb));
        self.bits += 1;
    }

    /// Record a join's mask difference.
    pub fn set_join_diff(&mut self, gid: GateId, diff: Label) {
        assert!(
            self.per_gate[gid].is_none(),
            "per-gate material already set for gate {}",
            gid
        );
        self.bits += label_bits(&diff);
        self.per_gate[gid] = Some(GateMaterial::JoinDiff(diff));
    }

    /// Append an output mask.
    pub fn push_output_mask(&mut self, mask: Label) {
        self.bits += label_bits(&mask);
        self.output_masks.push(mask);
    }

    /// Lookup per-gate material for a switch or join.
    pub fn material(&self, gid: GateId) -> Option<&GateMaterial> {
        self.per_gate[gid].as_ref()
    }

    /// Output masks in declaration order.
    pub fn output_masks(&self) -> &[Label] {
        &self.output_masks
    }

    /// Total bits of encoding.
    pub fn total_bits(&self) -> usize {
        self.bits
    }

    /// True if nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }
}

/// Bit cost of a single label.
pub fn label_bits(l: &Label) -> usize {
    match l {
        Label::Cf(c) => c.bit_len(),
        Label::Ncf(n) => {
            if n.modulus <= 1 {
                0
            } else {
                (n.modulus - 1).ilog2() as usize + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::label::{CfLabel, Label, NcfLabel};
    use super::*;

    #[test]
    fn test_empty() {
        let p = Program::with_num_gates(5);
        assert_eq!(p.total_bits(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_switch_lsb_one_bit() {
        let mut p = Program::with_num_gates(3);
        p.set_switch_lsb(1, true);
        assert_eq!(p.total_bits(), 1);
        assert!(matches!(
            p.material(1),
            Some(GateMaterial::SwitchLsb(true))
        ));
    }

    #[test]
    fn test_join_diff_size() {
        let mut p = Program::with_num_gates(2);
        p.set_join_diff(0, Label::Cf(CfLabel::zero(1 << 22)));
        assert_eq!(p.total_bits(), 128 * 22);
    }

    #[test]
    fn test_output_mask() {
        let mut p = Program::with_num_gates(0);
        p.push_output_mask(Label::Ncf(NcfLabel::zero(409)));
        assert_eq!(p.total_bits(), 9);
    }
}
