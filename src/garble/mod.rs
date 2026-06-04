//! Garbling and evaluation for switch systems.
//!
//! Entry points:
//! * [`garbler::garble`] — turn a `System` + input/output wire declarations into
//!   a garbled program.
//! * [`evaluator::eval`] — decode the garbled program at the evaluator's input.
//! * [`pipeline::Pipeline`] — streaming garble+eval that processes the
//!   computation phase by phase, dropping intermediate state at each boundary.
//!
//! Labels live in [`label`]: CF labels are bit-packed to `λ·k` bits so the
//! per-wire memory cost tracks the paper's bound exactly.

/// aarch64-specific intrinsics (NEON + AES) used by the CCRH.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
/// Evaluator: decode labels through gates.
pub mod evaluator;
/// Garbler: build masks, program, and input encoding.
pub mod garbler;
/// Correlation-robust hash (CCRND from Eprint 2019/074).
pub mod hash;
/// Label types (CF bit-packed + NCF single element).
pub mod label;
/// Streaming garble + eval pipeline.
pub mod pipeline;
/// Garbled program encoding.
pub mod program;

pub use evaluator::{eval, eval_with_labels};
pub use garbler::{garble, normalize_delta};
pub use label::{LAMBDA, Label};
pub use pipeline::{BatchCost, CarryId, CarryItem, PhaseStats, Pipeline, sample_cf_mask};
pub use program::Program;
