//! The garbling engine: the four straight-line steps that turn `x` into the
//! per-prime residues `a·(x mod p_i) + b`, and the machinery they share.
//!
//! Steps 1–3 are the **computational GC** (bits → one-hot of `x mod p_i`); step
//! 4 is the **information-theoretic GC** (one-hot → `a·x+b`). All four garble
//! and evaluate as straight-line loops over bare labels — the garbler's pass is
//! x-blind, the evaluator (which knows `x`) hashes only closed switches and
//! solves each open one through a join.
//!
//! * [`onehot`] — the two "moves" steps 1 & 2 are both built from: a one-hot
//!   doubling tree and width-`l` casts + a pin join, plus the bare-lane types.
//! * [`chunk`] — step 1: input bits → a ring word.
//! * [`extract`] — step 2: a ring word → its bits + a binary one-hot + upcast.
//! * [`fold`] — step 3: a binary one-hot → the length-`p_i` one-hot of `x mod p_i`.
//! * [`body`] — step 4: that one-hot → `a·(x mod p_i) + b`.

pub mod body;
pub mod chunk;
pub mod extract;
pub mod fold;
pub mod onehot;

/// The garbled-material + hash footprint of one step (for [`crate::affine::Stats`]).
/// A CF switch on Z_{2^w} costs `w` CCRH blocks, a CF join `w` λ-bit units; an
/// NCF switch/join costs one residue. Which side of the CF/NCF ledger a step's
/// cost lands on is decided by the driver, not the step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    /// Bits the garbler emits (the join diffs).
    pub program_bits: usize,
    /// Join width: `lg|G|` units for CF steps, bits for the NCF body.
    pub join_complexity: usize,
    /// CCRH blocks (garbler-side: every switch).
    pub hash_count: usize,
}

impl Cost {
    pub(crate) fn add(&mut self, o: Cost) {
        self.program_bits += o.program_bits;
        self.join_complexity += o.join_complexity;
        self.hash_count += o.hash_count;
    }
}
