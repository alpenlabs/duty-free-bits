// Crate lints
#![deny(rust_2018_idioms)]
#![deny(unused_crate_dependencies, unused_must_use)]
#![warn(missing_debug_implementations, unreachable_pub, missing_docs)]
#![warn(rustdoc::all)]
#![warn(clippy::too_long_first_doc_paragraph)]
// New default clippy lint (nightly ≥ 2026-07) that flags idiomatic
// `chunks_exact(CONST)` over the `as_chunks` split; the rewrite is a churny
// follow-up on hot loops, not worth gating this PR on.
#![allow(clippy::chunks_exact_to_as_chunks)]
//! Garbled evaluation of `a·x + b` over a CRT primorial, as a single
//! straight-line protocol — no gate graph, no constraint solver, no worklist.
//!
//! # One primitive: one-hot scaling
//!
//! Every wire value `v` is held as a **sharing of `v·Δ`** for a global secret
//! offset `Δ`: the garbler holds one share (its *mask*), the evaluator holds the
//! other, and their difference is exactly `v·Δ`. One party's share is a
//! **label**. A **one-hot vector** is a vector of labels sharing all-zero except
//! at a single *active position* — which the evaluator knows — where it
//! shares `1`.
//!
//! The entire computational GC is one move, applied over and over:
//!
//! > **one-hot scaling** — given a one-hot vector (sharing `1` at the active
//! > position), *one ciphertext* (sized to the scalar) turns it into a one-hot
//! > vector whose active position shares a chosen **scalar** and whose off
//! > positions share `0`.
//!
//! The evaluator, knowing the active position, opens the one hidden share from
//! that ciphertext and derives every off-position share locally. Scaling into a
//! *larger* ring is **upcasting**. And every *linear* map on labels or one-hot
//! vectors — adding, weighting by a constant, reindexing, summing into residue
//! classes, reducing mod `p`, reading off a bit — is **free**: no ciphertext,
//! just local share arithmetic. So the protocol is *repeated one-hot scaling +
//! free recombination*.
//!
//! # The pipeline
//!
//! A garbler holds the affine coefficients `(a, b)`; an evaluator holds the
//! input `x` in the clear, so it knows every active position. The garbler sends
//! the scaling ciphertexts; the evaluator decodes `a·x + b mod p_i` for each CRT
//! prime `p_i` and reconstructs `a·x + b` over the primorial. Four steps,
//! composed by [`affine::build_s_aff`]:
//!
//! ```text
//!   x bits ─▶ 1. CHUNK ─▶ ring words ─▶ 2. EXTRACT ─▶ bits + one-hot
//!                                                        │
//!            a·x+b mod p ◀─ 4. BODY ◀─ mod-p one-hot ◀─ 3. FOLD
//! ```
//!
//! 1. **chunk** ([`gc::chunk`]) — grow a one-hot over each group of `lg n` input
//!    bits and upcast it to a ring word `w_c ∈ Z_{2^ℓ}`.
//! 2. **extract** ([`gc::extract`]) — per prime, free-recombine the words into
//!    `r_i ≡ x (mod p_i)`, then decompose it (one-hot scaling + upcast) into
//!    sub-chunk bits + a one-hot of its low sub-chunk.
//! 3. **fold** ([`gc::fold`]) — free-reindex that one-hot into residue classes
//!    mod `p_i` and scale in the remaining bits, giving a length-`p_i` one-hot
//!    of `x mod p_i`.
//! 4. **body** ([`gc::body`]) — the canonical one-hot scaling: one ciphertext
//!    puts `a` at the active slot, then a free recombination reads out
//!    `a·(x mod p_i) + b`.
//!
//! Steps 1–3 are the **computational GC** (each scaling's ciphertext rests on
//! the CCRH); step 4 is the **information-theoretic GC** (each scaling is one
//! `Z_p` residue). See [`gc`].
//!
//! # Label flavours
//!
//! All three are sharings of `v·Δ`, differing only in the ring:
//!
//! * **boolean label** — XOR shares over `Z_2` (`v ∈ {0,1}`). A scaling's
//!   *control* must be a boolean label, so steps 1–3 build their one-hots from
//!   these. A boolean-label scaling costs λ bits.
//! * **arithmetic label** — additive shares over `Z_{2^k}`: what upcasting
//!   produces, and what a ring word is.
//! * **`Z_p` share** — plain additive shares of `v mod p` (the `Δ = 1` case).
//!   The body works here; its scaling costs one residue.
//!
//! The `_cf` / `_ncf` suffixes on the cost fields ([`affine::Stats`]) split the
//! ledger between the boolean-label steps (1–3) and the `Z_p`-share body (4).
//!
//! # Module map
//!
//! * [`label`] — the label (one share of `v·Δ`) and its ring arithmetic.
//! * [`crypto`] — the CCRH core (fixed-key AES) + nonce-freshness rules.
//! * [`hash`] — the label↔block wrapper over [`crypto`].
//! * [`gc`] — the four steps and the one-hot machinery they share.
//! * [`crt`] — CRT parameters and Garner reconstruction.
//! * [`affine`] — the driver ([`affine::build_s_aff`]) that composes the steps.
//! * [`bitdecomp`] — a bit-decomposition `a·x+b` baseline the benchmarks compare to.

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
