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
//! * [`system`] + [`garble`] + [`components::ohe`]/[`components::convert`] — the
//!   computational Yao GC engine (Phase 1: `bin(x)` → one-hot CRT).
//! * [`it_gc`] — the information-theoretic GC (Phase 3: one-hot → `a·x+b`).
//! * [`components::affine`] — composes the two via the streaming [`garble::Pipeline`].

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Higher-level circuit components (builders + the affine composition).
pub mod components;

/// CCRH core + nonce discipline (the bridge).
pub mod crypto;

/// Concrete cleartext execution of a system.
pub mod exec;

/// Garbling and evaluation (the computational GC engine).
pub mod garble;

/// Label-aware CCRH facade.
pub mod hash;

/// Information-theoretic GC: the per-prime body.
pub mod it_gc;

/// CF and NCF garbled-circuit labels.
pub mod label;

/// The constraint system: wires, gates, derived operations.
pub mod system;

/// Core types: values, wires, gates, ring arithmetic.
pub mod types;

#[cfg(test)]
mod tests;
