//! Garbled program encoding.
//!
//! Switches contribute **zero** communication: the evaluator knows the input `x`
//! in cleartext, so it derives every switch control itself (paper §3.3) — nothing
//! is revealed. The program therefore carries only:
//! * one **join diff** (`X_in0 − X_in1`) per `Join` gate, letting the evaluator
//!   translate between the two sides' masks.
//! * one **output mask** per declared NCF output wire.

use crate::label::Label;
use crate::types::GateId;

/// Garbled program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Join diff `X_in0 − X_in1` per gate id; `None` for non-join gates.
    join_diffs: Vec<Option<Label>>,
    /// One mask per declared output wire, in the caller's order.
    output_masks: Vec<Label>,
    /// Accumulated bit count.
    bits: usize,
}

impl Program {
    /// New program sized for `num_gates`.
    pub fn with_num_gates(num_gates: usize) -> Self {
        Self {
            join_diffs: vec![None; num_gates],
            output_masks: Vec::new(),
            bits: 0,
        }
    }

    /// Record a join's mask difference.
    pub fn set_join_diff(&mut self, gid: GateId, diff: Label) {
        assert!(
            self.join_diffs[gid].is_none(),
            "join diff already set for gate {}",
            gid
        );
        self.bits += label_bits(&diff);
        self.join_diffs[gid] = Some(diff);
    }

    /// Append an output mask.
    pub fn push_output_mask(&mut self, mask: Label) {
        self.bits += label_bits(&mask);
        self.output_masks.push(mask);
    }

    /// The join diff for a join gate (`None` if the gate emitted nothing).
    pub fn join_diff(&self, gid: GateId) -> Option<&Label> {
        self.join_diffs[gid].as_ref()
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
    use super::*;
    use crate::label::{CfLabel, Label, NcfLabel};

    #[test]
    fn test_empty() {
        let p = Program::with_num_gates(5);
        assert_eq!(p.total_bits(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_join_diff_size() {
        let mut p = Program::with_num_gates(2);
        p.set_join_diff(0, Label::Cf(CfLabel::zero(1 << 22)));
        assert_eq!(p.total_bits(), 128 * 22);
        assert!(p.join_diff(0).is_some());
        assert!(p.join_diff(1).is_none());
    }

    #[test]
    fn test_output_mask() {
        let mut p = Program::with_num_gates(0);
        p.push_output_mask(Label::Ncf(NcfLabel::zero(409)));
        assert_eq!(p.total_bits(), 9);
    }
}
