//! Step 3 of 4 (computational GC): reduce the extract step's boolean one-hot to
//! a length-`p_i` one-hot of `x mod p_i` that [`super::body`] consumes.
//!
//! Free-reindex the one-hot into residue classes mod `p_i` (h[i mod p] = Σ, a
//! free Z₂ recombine), then one-hot-scale in the remaining sub-chunk bits — one
//! scaling per fold bit, hashing all `p` slots. Straight-line garble/eval over
//! boolean labels.
//!
//! The fold is regular — per folded bit `b`, per slot `r`:
//!
//! ```text
//!   h'[r]    = scale(h[r])                X_h'[r] = H(X_h[r])   (one CCRH block)
//!   s        = Σ_r h'[r]                  X_s     = ⊕_r X_h'[r]
//!   scale(s, bit)                         diff    = X_s ⊕ X_bit (communicated)
//!   h_new[r] = h[r] + h'[r] + h'[src]     src     = (r − 2^pos mod p) mod p
//! ```
//!
//! Every wire is a boolean label, so labels are bare `[u64; 2]` blocks and ring
//! ops are XOR. The garbler is fully forward. The evaluator holds `r` in the
//! clear, so it knows every active position; at the single *hot* slot — where
//! the scaling is opened — it recovers the pad backward through the scaling:
//! `L_h'[hot] = L_s ⊕ (⊕_{r≠hot} L_h'[r])` with `L_s = L_bit ⊕ diff` — the
//! opened scaling's backward chain, telescoped.
//!
//! Scaling hashes draw fresh bulk-domain CCRH nonces from a caller-supplied
//! window of `bits.len()·p` ids (see [`crate::crypto::nonce`]); the garbler and
//! evaluator use the same window, so pads agree at every non-hot slot.

use crate::crt::pow2_mod;
use crate::gc::Cost;
use crate::hash;
use crate::label::Label;

/// Garbler-side output of one fold.
#[derive(Debug)]
pub struct FoldGarbleOutput {
    /// Masks of the final length-`p` one-hot (boolean labels).
    pub h_p_masks: Vec<Label>,
    /// The one residue each scaling communicates, per folded bit
    /// (`X_s ⊕ X_bit`, boolean label).
    pub join_diffs: Vec<Label>,
    /// Footprint for telemetry.
    pub cost: Cost,
}

/// Packed words of a boolean label.
#[inline]
fn z2_words(l: &Label) -> [u64; 2] {
    debug_assert_eq!(l.modulus(), 2, "fold wires are boolean labels (Z_2)");
    let raw = l.raw_bits();
    [raw[0], raw[1]]
}

#[inline]
fn z2_label(w: [u64; 2]) -> Label {
    Label::from_raw_bits(vec![w[0], w[1]], 2)
}

/// Garbler side: fold the first sub-chunk's boolean one-hot into a mod-`p` one-hot.
///
/// `first_bin_hot_masks` are the masks of the boolean one-hot of the low
/// `first_width` bits; `bit_masks` the masks of bit positions `first_width..`.
/// `nonce_base` is the start of a fresh window of `bit_masks.len()·p`
/// bulk-domain CCRH ids.
pub fn fold_batch_garble(
    p: u64,
    first_bin_hot_masks: &[Label],
    bit_masks: &[Label],
    first_width: u32,
    nonce_base: u64,
) -> FoldGarbleOutput {
    let p_usize = p as usize;

    // h init: free reindex into residue classes — h[r'] = Σ_{i ≡ r' (mod p)} first_bin_hot[i].
    let mut h = vec![[0u64; 2]; p_usize];
    for (i, m) in first_bin_hot_masks.iter().enumerate() {
        let w = z2_words(m);
        let slot = &mut h[i % p_usize];
        slot[0] ^= w[0];
        slot[1] ^= w[1];
    }

    let mut join_diffs = Vec::with_capacity(bit_masks.len());
    let mut h_prime = vec![[0u64; 2]; p_usize];
    for (b_idx, bm) in bit_masks.iter().enumerate() {
        let shift = pow2_mod(first_width + b_idx as u32, p) as usize;
        let bit_nonce_base = nonce_base + (b_idx * p_usize) as u64;

        // Pads + their running sum: X_h'[r] = H(X_h[r]), X_s = ⊕ X_h'[r].
        let mut s = [0u64; 2];
        for (r, hp) in h_prime.iter_mut().enumerate() {
            *hp = hash::hash_z2(&h[r], bit_nonce_base + r as u64);
            s[0] ^= hp[0];
            s[1] ^= hp[1];
        }

        // The one residue communicated per bit: diff = X_s ⊕ X_bit.
        let bit_words = z2_words(bm);
        join_diffs.push(z2_label([s[0] ^ bit_words[0], s[1] ^ bit_words[1]]));

        // h[r] ⊕= h'[r] ⊕ h'[src] — in place: the update at r reads only
        // h[r] itself and h_prime.
        for r in 0..p_usize {
            let src = (r + p_usize - shift) % p_usize;
            h[r][0] ^= h_prime[r][0] ^ h_prime[src][0];
            h[r][1] ^= h_prime[r][1] ^ h_prime[src][1];
        }
    }

    FoldGarbleOutput {
        h_p_masks: h.into_iter().map(z2_label).collect(),
        join_diffs,
        cost: fold_cost(p, bit_masks.len()),
    }
}

/// Footprint of folding `bits` bit positions mod `p` (per side; each scaling
/// is charged once).
fn fold_cost(p: u64, bits: usize) -> Cost {
    Cost {
        program_bits: bits * crate::label::LAMBDA,
        join_complexity: bits,
        hash_count: bits * p as usize,
    }
}

/// Evaluator side: the label-side mirror of [`fold_batch_garble`].
///
/// `r` is the cleartext value whose one-hot is being folded (the evaluator
/// holds `x`, hence `r`): bit values and every intermediate active position
/// derive from it. Returns the final one-hot labels; slot `r mod p` is hot.
pub fn fold_batch_eval(
    p: u64,
    r: u64,
    first_bin_hot_labels: &[Label],
    bit_labels: &[Label],
    join_diffs: &[Label],
    first_width: u32,
    nonce_base: u64,
) -> Vec<Label> {
    let p_usize = p as usize;
    assert_eq!(bit_labels.len(), join_diffs.len());
    assert_eq!(first_bin_hot_labels.len(), 1usize << first_width);

    let mut h = vec![[0u64; 2]; p_usize];
    for (i, l) in first_bin_hot_labels.iter().enumerate() {
        let w = z2_words(l);
        let slot = &mut h[i % p_usize];
        slot[0] ^= w[0];
        slot[1] ^= w[1];
    }
    // Running hot position: starts at (r mod 2^first_width) mod p.
    let mut hot = ((r & ((1u64 << first_width) - 1)) % p) as usize;

    let mut h_prime = vec![[0u64; 2]; p_usize];
    for (b_idx, (bl, diff)) in bit_labels.iter().zip(join_diffs).enumerate() {
        let pos = first_width + b_idx as u32;
        let shift = pow2_mod(pos, p) as usize;
        let bit_nonce_base = nonce_base + (b_idx * p_usize) as u64;

        // Non-hot slots are off (share 0), so the pad recomputes exactly.
        let mut s_known = [0u64; 2];
        for (rr, hp) in h_prime.iter_mut().enumerate() {
            if rr == hot {
                continue;
            }
            *hp = hash::hash_z2(&h[rr], bit_nonce_base + rr as u64);
            s_known[0] ^= hp[0];
            s_known[1] ^= hp[1];
        }
        // Hot slot, backward through the scaling: L_s = L_bit ⊕ diff, then
        // L_h'[hot] = L_s ⊕ (⊕_{r≠hot} L_h'[r]).
        let bit_words = z2_words(bl);
        let diff_words = z2_words(diff);
        h_prime[hot] = [
            bit_words[0] ^ diff_words[0] ^ s_known[0],
            bit_words[1] ^ diff_words[1] ^ s_known[1],
        ];

        for rr in 0..p_usize {
            let src = (rr + p_usize - shift) % p_usize;
            h[rr][0] ^= h_prime[rr][0] ^ h_prime[src][0];
            h[rr][1] ^= h_prime[rr][1] ^ h_prime[src][1];
        }
        // Value-side: folding bit b moves the hot slot by bit·2^pos.
        if (r >> pos) & 1 == 1 {
            hot = (hot + shift) % p_usize;
        }
    }
    debug_assert_eq!(hot as u64, r % p, "hot tracking diverged");

    h.into_iter().map(z2_label).collect()
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

    /// labels = masks + value·Δ₂ in: garble + eval a fold and check the final
    /// one-hot satisfies the carry invariant with exactly slot `r mod p` hot.
    #[test]
    fn test_fold_label_mask_invariant() {
        let mut rng = rand::rng();
        let delta: u128 = rng.random();
        let d2 = label::delta_r(delta, 2);

        for &p in &[2u64, 3, 5, 7, 11, 29] {
            let first_width = 4u32;
            let fold_bits = 3usize; // value space: 7 bits
            for _ in 0..4 {
                let r: u64 = rng.random_range(0..1 << (first_width as usize + fold_bits));

                // Seed first_bin_hot as a garbled one-hot of (r mod 2^w0):
                // mask random, label = mask + bit·Δ₂.
                let low = (r & ((1 << first_width) - 1)) as usize;
                let masks: Vec<Label> = (0..1 << first_width).map(|_| rand_z2(&mut rng)).collect();
                let labels: Vec<Label> = masks
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        if i == low {
                            label::add(m, &d2)
                        } else {
                            m.clone()
                        }
                    })
                    .collect();
                // Fold-bit wires likewise.
                let bit_masks: Vec<Label> = (0..fold_bits).map(|_| rand_z2(&mut rng)).collect();
                let bit_labels: Vec<Label> = bit_masks
                    .iter()
                    .enumerate()
                    .map(|(j, m)| {
                        if (r >> (first_width + j as u32)) & 1 == 1 {
                            label::add(m, &d2)
                        } else {
                            m.clone()
                        }
                    })
                    .collect();

                let g = fold_batch_garble(p, &masks, &bit_masks, first_width, 7000);
                let h_p =
                    fold_batch_eval(p, r, &labels, &bit_labels, &g.join_diffs, first_width, 7000);

                let hot = (r % p) as usize;
                for (slot, (l, m)) in h_p.iter().zip(&g.h_p_masks).enumerate() {
                    let expected = if slot == hot {
                        label::add(m, &d2)
                    } else {
                        m.clone()
                    };
                    assert_eq!(
                        *l, expected,
                        "p={p}, r={r}, slot={slot}: label ≠ mask + v·Δ₂"
                    );
                }
            }
        }
    }
}
