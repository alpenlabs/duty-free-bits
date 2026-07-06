// Crate lints
#![deny(rust_2018_idioms)]
#![deny(unused_crate_dependencies, unused_must_use)]
#![warn(missing_debug_implementations, unreachable_pub, missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::too_long_first_doc_paragraph)]
// WIP branch: the module-map doc links reference an in-flux layout that the
// follow-up straight-line PR rewrites; don't gate this transient state on them.
#![allow(rustdoc::broken_intra_doc_links)]
//! Switch system framework for Duty Free Bits.
//!
//! Structured as a computational GC composed with an information-theoretic GC.
//! * [`label`] — CF (computational, λ-fold) and NCF (IT, single element) labels.
//! * [`crypto`] — the CCRH core + nonce rules.
//! * [`hash`] — the label-aware CCRH built atop `crypto`.
//! * [`system`] + [`comp_gc`] — the computational Yao GC engine (Phase 1: `bin(x)` → hot(CRT(x))).
//! * [`it_gc`] — the information-theoretic GC (Phase 3: hot(CRT(x)) → `a·x+b`).
//! * [`affine`] — composes the two via the streaming [`pipeline`].

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod crypto;

/// The S_aff affine-map driver (composes comp_gc + it_gc via the pipeline).
pub mod affine;

/// Computational Yao garbling (Phase 1).
pub mod comp_gc;

/// CRT parameters, reconstruction, and big-integer math.
pub mod crt;

/// Concrete cleartext execution of a system.
pub mod exec;

/// CCRH core + nonce rules.
pub mod hash;

/// Information-theoretic GC (Phase 3).
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
