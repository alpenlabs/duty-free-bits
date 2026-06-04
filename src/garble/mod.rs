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

/// Evaluator: decode labels through gates.
pub mod evaluator;
/// Garbler: build masks, program, and input encoding.
pub mod garbler;
/// Streaming garble + eval pipeline.
pub mod pipeline;
/// Garbled program encoding.
pub mod program;

// `label` (leaf) and `hash` (facade) now live at the crate root; re-export here
// so existing `garble::label` / `garble::hash` / `garble::Label` paths keep working.
pub use crate::hash;
pub use crate::label;

pub use crate::label::{LAMBDA, Label};
pub use crate::it_gc::BatchCost;
pub use evaluator::{eval, eval_with_labels};
pub use garbler::{garble, normalize_delta};
pub use pipeline::{CarryId, CarryItem, PhaseStats, Pipeline, sample_cf_mask};
pub use program::Program;
