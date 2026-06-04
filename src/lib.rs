// Crate lints (kept in source so the modern `[lints]` Cargo table doesn't trip
// the editor's stale schema). The `rust_2018_idioms` group is denied first so a
// later specific lint can still override it.
#![deny(rust_2018_idioms)]
#![deny(unused_crate_dependencies, unused_must_use)]
#![warn(missing_debug_implementations, unreachable_pub, missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::too_long_first_doc_paragraph)]
//! Switch system framework for garbled arithmetic circuits.
//!
//! Layered to mirror the protocol — a computational GC composed with an
//! information-theoretic GC across a CCRH bridge:
//! * [`label`] — CF (computational, λ-fold) and NCF (IT, single element) labels. Leaf.
//! * [`crypto`] — the Label-free CCRH core + nonce discipline (the bridge). Leaf.
//! * [`hash`] — the label-aware CCRH facade over `crypto`.
//! * [`system`] + [`comp_gc`] (with builders [`comp_gc::ohe`]/[`comp_gc::convert`])
//!   — the computational Yao GC engine (Phase 1: `bin(x)` → one-hot CRT).
//! * [`it_gc`] — the information-theoretic GC (Phase 3: one-hot → `a·x+b`).
//! * [`affine`] — composes the two via the streaming [`pipeline::Pipeline`].

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The S_aff affine-map driver (composes comp_gc + it_gc via the pipeline).
pub mod affine;

/// Computational Yao GC over the gate System (Phase 1: bin(x) -> OHE-CRT).
pub mod comp_gc;

/// CRT parameters, reconstruction, and big-integer math (leaf).
pub mod crt;

/// CCRH core + nonce discipline (the bridge).
pub mod crypto;

/// Concrete cleartext execution of a system.
pub mod exec;

/// Label-aware CCRH facade.
pub mod hash;

/// Information-theoretic GC: the per-prime body.
pub mod it_gc;

/// CF and NCF garbled-circuit labels.
pub mod label;

/// Streaming garble+eval orchestration.
pub mod pipeline;

/// The constraint system: wires, gates, derived operations.
pub mod system;

/// Core types: values, wires, gates, ring arithmetic.
pub mod types;

#[cfg(test)]
mod tests;
