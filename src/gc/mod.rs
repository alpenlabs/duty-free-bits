//! The garbling engine: the four straight-line steps that turn `x` into the
//! per-prime residues `a·(x mod p_i) + b`, and the machinery they share.
//!
//! Every step is the same move — **one-hot scaling** — plus free recombination
//! (see the crate docs). Steps 1–3 are the **computational GC** (input bits →
//! a one-hot of `x mod p_i`, each scaling costing one CCRH ciphertext); step 4
//! is the **information-theoretic GC** (one-hot → `a·x+b`, each scaling one
//! `Z_p` residue). All four garble and evaluate as straight-line loops: the
//! garbler is x-blind — it emits every scaling ciphertext — while the evaluator
//! knows `x`, hence every active position, so it opens only the one hidden share
//! per scaling and derives the rest for free.
//!
//! * `onehot` (crate-internal) — the shared moves steps 1 & 2 build on: growing
//!   a one-hot one bit at a time, upcasting its leaves, and the root scaling that
//!   delivers the active leaf's value, plus the bare-word working types.
//! * [`chunk`] — step 1: input bits → a ring word.
//! * [`extract`] — step 2: a ring word → its bits + a one-hot + upcast.
//! * [`fold`] — step 3: that one-hot → the length-`p_i` one-hot of `x mod p_i`.
//! * [`body`] — step 4: that one-hot → `a·(x mod p_i) + b`.

pub mod body;
pub mod chunk;
pub mod extract;
pub mod fold;
pub(crate) mod onehot;

/// The ciphertext + hash footprint of one step (for [`crate::affine::Stats`]).
/// A boolean-label scaling into `Z_{2^w}` costs `w` CCRH blocks and `w` λ-bit
/// units of communication; a `Z_p`-share scaling costs one residue. Which side
/// of the ledger a step's cost lands on is set by the driver, not the step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    /// Bits the garbler emits (the scaling ciphertexts).
    pub program_bits: usize,
    /// Scaling width: `lg|G|` units for the boolean-label steps, bits for the
    /// `Z_p`-share body.
    pub join_complexity: usize,
    /// CCRH blocks (the garbler is x-blind, so it hashes every slot of every scaling).
    pub hash_count: usize,
}

impl Cost {
    pub(crate) fn add(&mut self, o: Cost) {
        self.program_bits += o.program_bits;
        self.join_complexity += o.join_complexity;
        self.hash_count += o.hash_count;
    }
}
