//! Garbling and evaluation for switch systems.
//!
//! Entry points:
//! * [`garbler::garble`] — turn a `System` + input/output wire declarations into
//!   a garbled program.
//! * [`evaluator::eval`] — decode the garbled program at the evaluator's input.
//!
//! Labels live in [`label`]: CF labels are bit-packed to `λ·k` bits so the
//! per-wire memory cost tracks the paper's bound exactly.

/// Evaluator: decode labels through gates.
pub mod evaluator;
/// Garbler: build masks, program, and input encoding.
pub mod garbler;
/// Non-cryptographic CCRH stub.
pub mod hash;
/// Label types (CF bit-packed + NCF single element).
pub mod label;
/// Garbled program encoding.
pub mod program;

pub use evaluator::eval;
pub use garbler::{garble, normalize_delta};
pub use label::{LAMBDA, Label};
pub use program::Program;
