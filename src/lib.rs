// Crate lints
#![deny(rust_2018_idioms)]
#![deny(unused_crate_dependencies, unused_must_use)]
#![warn(missing_debug_implementations, unreachable_pub, missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::too_long_first_doc_paragraph)]
//! Garbled evaluation of `a·x + b` over a CRT primorial, as a single
//! straight-line protocol (no gate graph, no worklist).
//!
//! * [`label`] — the garbled label (λ coordinates over a power-of-two ring).
//! * [`crypto`] — the CCRH core + nonce rules.
//! * [`hash`] — the label-aware CCRH built atop `crypto`.
//! * [`comp_gc`] — the computational GC steps (Phase 1: `bin(x)` → hot(CRT(x))):
//!   chunk conversion, sub-chunk extraction, and the mod-p fold.
//! * [`it_gc`] — the information-theoretic GC (Phase 3: hot(CRT(x)) → `a·x+b`).
//! * [`affine`] — composes the steps into the production driver
//!   ([`affine::build_s_aff`]).
//! * [`bitdecomp`] — standalone switch-free `a·x+b` baseline for hash-count comparison.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod crypto;

pub mod affine;

pub mod bitdecomp;

pub mod comp_gc;

pub mod crt;

pub mod hash;

pub mod it_gc;

pub mod label;

#[cfg(test)]
mod tests;
