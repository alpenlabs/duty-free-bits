// Crate lints
#![deny(rust_2018_idioms)]
#![deny(unused_crate_dependencies, unused_must_use)]
#![warn(missing_debug_implementations, unreachable_pub, missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::too_long_first_doc_paragraph)]
//! Garbled evaluation of `a·x + b` over a CRT primorial, as a single
//! straight-line protocol — no gate graph, no constraint solver, no worklist.
//!
//! # The pipeline
//!
//! A garbler holds the affine coefficients `(a, b)`; an evaluator holds the
//! input `x` (in the clear — the *switch-private / data-public* setting). The
//! garbler sends garbled material; the evaluator decodes `a·x + b mod p_i` for
//! each CRT prime `p_i` and reconstructs `a·x + b` over the primorial. The
//! computation is four straight-line steps, composed by [`affine::build_s_aff`]:
//!
//! ```text
//!   x bits ─▶ 1. CHUNK ─▶ ring words ─▶ 2. EXTRACT ─▶ bits + one-hot
//!                                                        │
//!            a·x+b mod p ◀─ 4. BODY ◀─ mod-p one-hot ◀─ 3. FOLD
//! ```
//!
//! 1. **chunk** ([`gc::chunk`]) — pack `lg n` input bits into a ring word
//!    `w_c ∈ Z_{2^ℓ}`.
//! 2. **extract** ([`gc::extract`]) — per prime, form `r_i ≡ x (mod p_i)` from
//!    the words, then decompose it into sub-chunk bits + a binary one-hot.
//! 3. **fold** ([`gc::fold`]) — reduce that one-hot mod `p_i` to a length-`p_i`
//!    one-hot of `x mod p_i`.
//! 4. **body** ([`gc::body`]) — the information-theoretic GC: deliver
//!    `a·(x mod p_i) + b` from the one-hot.
//!
//! Steps 1–3 are the **computational GC** (its security rests on the CCRH);
//! step 4 is the **information-theoretic GC**. See [`gc`].
//!
//! # Two label domains (CF / NCF)
//!
//! A wire's two labels always differ by `value · Δ` for the global offset `Δ`
//! (free-XOR). The crate uses that in two representations, tagged throughout:
//!
//! * **CF** = *control-friendly*: a [`label::Label`] — λ coordinates over a
//!   power-of-two ring `Z_{2^k}`. The computational-GC steps (1–3) use these,
//!   because a switch's control must be a CF Z₂ wire. A CF join costs λ bits.
//! * **NCF** = *non-control-friendly*: a single `Z_p` residue (a bare `u64`,
//!   its own label). The IT body (step 4) works in this domain; an NCF join
//!   costs one residue.
//!
//! The `_cf` / `_ncf` suffixes on the cost fields ([`affine::Stats`]) split the
//! ledger along exactly this line.
//!
//! # Module map
//!
//! * [`label`] — the garbled label and its ring arithmetic.
//! * [`crypto`] — the CCRH core (fixed-key AES) + nonce-freshness rules.
//! * [`hash`] — the label↔block wrapper over [`crypto`].
//! * [`gc`] — the four garbling steps and the one-hot machinery they share.
//! * [`crt`] — CRT parameters and Garner reconstruction.
//! * [`affine`] — the driver ([`affine::build_s_aff`]) that composes the steps.
//! * [`bitdecomp`] — a switch-free `a·x+b` baseline the benchmarks compare to.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod crypto;

pub mod affine;

pub mod bitdecomp;

pub mod crt;

pub mod gc;

pub mod hash;

pub mod label;

#[cfg(test)]
mod tests;
