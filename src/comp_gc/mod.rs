//! Computational Yao garbling over the gate `System` (Phase 1: bin(x) -> one-hot CRT).
//!
//! * [`ohe`] / [`convert`] build the Phase-1 circuits.
//! * `arena` (crate-private) holds the compiled garbler and the fused
//!   value+label evaluator over flat label storage — the path keyed all-CF
//!   phases take.
//! * [`garbler`] / [`evaluator`] are the worklist garbler and journal-replay
//!   evaluator: they garble/evaluate unkeyed (and NCF) phases and record each
//!   shape's tape for the compiled path.
//! * [`fold`] is the straight-line kernel for the mod-p OHE fold.
//! * [`program`] is the garbled program encoding (join diffs + output masks).

pub(crate) mod arena;
pub mod convert;
pub mod evaluator;
pub mod extract;
pub mod fold;
pub mod garbler;
pub mod ohe;
pub mod program;

pub use evaluator::replay_with_labels;
pub use garbler::garble;
pub use program::Program;
