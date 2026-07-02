//! Phase 1 (computational GC): binary input → one-hot of `x mod p_i`.
//!
//! Straight-line garble/eval over bare labels, no gate graph:
//! * [`extract`] builds the chunk words and decomposes them into bits, a
//!   binary one-hot, and the peel upcast (the fused `word_to_bin_up`).
//! * [`fold`] reduces the first sub-chunk's one-hot mod `p_i` and folds in the
//!   remaining bits.

pub mod extract;
pub mod fold;
