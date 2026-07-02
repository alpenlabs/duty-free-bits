//! The garbling engine: the four straight-line steps that turn `x` into the
//! per-prime residues `a·(x mod p_i) + b`, and the machinery they share.
//!
//! Steps 1–3 are the **computational GC** (bits → one-hot of `x mod p_i`); step
//! 4 is the **information-theoretic GC** (one-hot → `a·x+b`). All four garble
//! and evaluate as straight-line loops over bare labels — the garbler's pass is
//! x-blind, the evaluator (which knows `x`) hashes only closed switches and
//! solves each open one through a join.
//!
//! * [`onehot`] — the two "moves" steps 1 & 2 are both built from: a one-hot
//!   doubling tree and width-`l` casts + a pin join, plus the bare-lane types.
//! * [`chunk`] — step 1: input bits → a ring word.
//! * [`extract`] — step 2: a ring word → its bits + a binary one-hot + upcast.
//! * [`fold`] — step 3: a binary one-hot → the length-`p_i` one-hot of `x mod p_i`.
//! * [`body`] — step 4: that one-hot → `a·(x mod p_i) + b`.

pub mod body;
pub mod chunk;
pub mod extract;
pub mod fold;
pub mod onehot;
