//! Computational Yao garbling over the gate `System` (Phase 1: bin(x) -> one-hot CRT).
//!
//! * [`garbler::garble`] / [`evaluator::eval_with_labels`] garble + decode a System.
//! * [`ohe`] / [`convert`] are the Phase-1 builders.

/// Conversions between binary, word, and ring representations (Phase-1 builder).
pub mod convert;
pub mod evaluator;
pub mod garbler;
/// One-hot encoding and scaling (Phase-1 builder).
pub mod ohe;
pub mod program;

pub use evaluator::eval_with_labels;
pub use garbler::{garble, normalize_delta};
pub use program::Program;
