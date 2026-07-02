//! Step 1 / 4 (computational GC): pack `lg n` input bits into a ring word
//! `w_c ∈ Z_{2^ℓ}` — the straight-line `bin_to_word`.
//!
//! Inputs: the `n` input-bit labels of one chunk.  Output: the chunk word `w_c`
//! (its garbled label). Built on the one-hot tree + casts of [`super::onehot`]:
//! a one-hot over the chunk's bits, width-ℓ casts of the leaves, and
//! `word = Σ_p p·A_p`. The pin join lets the evaluator recover the one hot cast
//! it cannot derive.

use super::Cost;
use super::onehot::*;
use crate::hash;
use crate::label::{LAMBDA, Label, delta_r};


/// (bulk ids, solo ids) consumed by one chunk step over `n_bits` input bits.
pub fn chunk_nonces(n_bits: u32) -> (u64, u64) {
    (tree_ids(n_bits), 1u64 << n_bits)
}

/// Garbler-side output of one chunk-conversion step.
#[derive(Debug)]
pub struct ChunkGarbleOutput {
    /// Mask of the chunk word (CF Z_{2^ell}).
    pub word_mask: Label,
    /// The n−1 scale-join diffs and the width-ell pin diff.
    pub scale: Vec<Z2>,
    /// Pin-join diff (width ell).
    pub pin: Wide,
    /// Ledger footprint.
    pub cost: Cost,
}

/// Garbler side: pack `bits` (CF Z₂ masks, LSB-first) into a Z_{2^ell}
/// word — the straight-line `bin_to_word` (one-hot tree over the input bits,
/// width-ell casts, `Σ p·A_p`).
pub fn chunk_batch_garble(
    bit_masks: &[Label],
    ell: u32,
    delta: u128,
    nonce_bulk: u64,
    nonce_solo: u64,
) -> ChunkGarbleOutput {
    let nb = bit_masks.len() as u32;
    assert!(
        (2..=31).contains(&ell) && ell >= nb,
        "chunk kernel requires 2 <= nb <= ell <= 31 (got nb={nb}, ell={ell})"
    );
    let mask_l = ((1u64 << ell) - 1) as u32;
    let d2 = {
        let d = delta_r(delta, 2);
        let raw = d.raw_bits();
        [raw[0], raw[1]]
    };
    let b0 = z2_of(&bit_masks[0]);
    let lvl1 = [[b0[0] ^ d2[0], b0[1] ^ d2[1]], b0];
    let (leaves, ysums) = tree_garble(nb, lvl1, nonce_bulk);

    let n = 1usize << nb;
    let mut casts = vec![[0u32; LAMBDA]; n];
    for (leaf, cast) in casts.iter_mut().enumerate() {
        hash_cast(&leaves[leaf], nonce_solo + leaf as u64, ell, cast);
    }
    let (chain, root) = peel_chain(casts);
    let word_mask = chain[nb as usize - 1]; // Σ p·A_p

    let scale: Vec<Z2> = ysums
        .iter()
        .zip(bit_masks.iter().skip(1))
        .map(|(y, b)| {
            let bw = z2_of(b);
            [y[0] ^ bw[0], y[1] ^ bw[1]]
        })
        .collect();
    let mut pin = root;
    let dl = delta_r(delta, 1u64 << ell);
    wide_add(&mut pin, &wide_of(&dl));
    for lane in pin.iter_mut() {
        *lane &= mask_l;
    }

    ChunkGarbleOutput {
        word_mask: wide_label(&word_mask, 1u64 << ell),
        scale,
        pin,
        cost: stage_cost(nb, ell),
    }
}

/// Evaluator side: the label-side mirror of [`chunk_batch_garble`].
/// `value` is the chunk's cleartext value (the evaluator knows its bits).
pub fn chunk_batch_eval(
    bit_labels: &[Label],
    value: u64,
    ell: u32,
    g_scale: &[Z2],
    g_pin: &Wide,
    nonce_bulk: u64,
    nonce_solo: u64,
) -> Label {
    let nb = bit_labels.len() as u32;
    assert!(
        (2..=31).contains(&ell) && ell >= nb,
        "chunk kernel requires 2 <= nb <= ell <= 31 (got nb={nb}, ell={ell})"
    );

    // Tree: expand every level; the hot slot solves through the scale join.
    let b0 = z2_of(&bit_labels[0]);
    let mut lvl: Vec<Z2> = vec![b0, b0];
    for m in 1..nb {
        let width = 1usize << m;
        let base = nonce_bulk + ((1u64 << m) - 2);
        let hot = (value & ((1u64 << m) - 1)) as usize;
        let mut next = vec![[0u64; 2]; width * 2];
        let mut ysum = [0u64; 2];
        for (j, bj) in lvl.iter().enumerate() {
            if j == hot {
                continue;
            }
            let y = hash::hash_z2(bj, base + j as u64);
            ysum[0] ^= y[0];
            ysum[1] ^= y[1];
            next[j] = [bj[0] ^ y[0], bj[1] ^ y[1]];
            next[j + width] = y;
        }
        let bm = z2_of(&bit_labels[m as usize]);
        let d = g_scale[m as usize - 1];
        let y_hot = [bm[0] ^ d[0] ^ ysum[0], bm[1] ^ d[1] ^ ysum[1]];
        next[hot] = [lvl[hot][0] ^ y_hot[0], lvl[hot][1] ^ y_hot[1]];
        next[hot + width] = y_hot;
        lvl = next;
    }

    // Casts of the non-hot leaves; the hot cast solves through the pin join,
    // and the word label is Σ p·A_p with the hot term substituted.
    let hot = (value & ((1u64 << nb) - 1)) as usize;
    let mut word = [0u32; LAMBDA];
    let mut t_sum = [0u32; LAMBDA];
    let mut cast = [0u32; LAMBDA];
    for (slot, leaf) in lvl.iter().enumerate() {
        if slot == hot {
            continue;
        }
        hash_cast(leaf, nonce_solo + slot as u64, ell, &mut cast);
        wide_madd(&mut word, slot as u32, &cast);
        wide_add(&mut t_sum, &cast);
    }
    let mut hot_cast = *g_pin; // L_root = pin diff (constant-1 label is 0)
    wide_sub(&mut hot_cast, &t_sum);
    wide_madd(&mut word, hot as u32, &hot_cast);

    wide_label(&word, 1u64 << ell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{self, LAMBDA};
    use rand::Rng;

    fn rand_z2(rng: &mut impl Rng) -> Label {
        let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..2u64)).collect();
        Label::from_coords(&coords, 2)
    }

    /// Chunk step: word label = word mask + v·Δ for every value, at the
    /// production shape (nb = 8, ell = 22) and a small one.
    #[test]
    fn test_chunk_label_mask_invariant_production() {
        chunk_invariant_shape(8, 22);
    }

    #[test]
    fn test_chunk_label_mask_invariant() {
        chunk_invariant_shape(4, 12);
    }

    fn chunk_invariant_shape(nb: u32, ell: u32) {
        let mut rng = rand::rng();
        let delta: u128 = rng.random::<u128>() | 1;
        let d2 = delta_r(delta, 2);
        let dl = delta_r(delta, 1u64 << ell);
    
        let bit_masks: Vec<Label> = (0..nb).map(|_| rand_z2(&mut rng)).collect();
        let g = chunk_batch_garble(&bit_masks, ell, delta, 5_000, 9_000);
        let lim = 1u32 << ell;
        assert!(g.pin.iter().all(|&lane| lane < lim), "pin lane not canonical");
        assert_eq!(g.cost.join_complexity as u32, nb - 1 + ell);
        assert_eq!(
            g.cost.hash_count as u64,
            (1u64 << nb) - 2 + ell as u64 * (1u64 << nb)
        );
    
        for v in 0..(1u64 << nb) {
            let bit_labels: Vec<Label> = bit_masks
                .iter()
                .enumerate()
                .map(|(j, m)| {
                    if (v >> j) & 1 == 1 {
                        label::add(m, &d2)
                    } else {
                        m.clone()
                    }
                })
                .collect();
            let w = chunk_batch_eval(&bit_labels, v, ell, &g.scale, &g.pin, 5_000, 9_000);
            let expect = label::add(&g.word_mask, &label::scalar_mul(v, &dl));
            assert_eq!(w, expect, "chunk word label ≠ mask + v·Δ for v={v} nb={nb} ell={ell}");
        }
    }
}
