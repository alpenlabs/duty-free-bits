//! Computational Yao garbling over the gate `System` (Phase 1: bin(x) -> one-hot CRT).
//!
//! * [`ohe`] / [`convert`] build the Phase-1 circuits.
//! * `arena` (crate-private) holds the production engines: compiled
//!   garbling and fused
//!   value+label evaluation over flat label storage.
//! * [`garbler`] / [`evaluator`] are the `Label`-path worklist engines —
//!   first-of-shape recording, fallback for shapes the arena cannot host,
//!   and the checked references for the differential tests.
//! * [`fold`] is the straight-line kernel for the mod-p OHE fold.
//! * [`program`] is the garbled program encoding (join diffs + output masks).

pub(crate) mod arena;
pub mod convert;
pub mod evaluator;
pub mod fold;
pub mod garbler;
pub mod ohe;
pub mod program;

pub use evaluator::{eval_with_labels, replay_with_labels};
pub use garbler::garble;
pub use program::Program;
