//! Information-theoretic GC: deliver `a·(x mod p_i) + b` per prime.
//!
//! Phase 1 handed us a binary one-hot `h_p` of `hot = x mod p_i`: `p_i` CF Z_2
//! wires, exactly the one at index `hot` carrying a 1. Hashing wire `i`'s label
//! gives a fresh one-time pad `pad_i = H(h_p[i]) ∈ Z_{p_i}` — one pad per slot.
//!
//! **One pad is hidden from the evaluator.** The evaluator knows `x`, hence
//! `hot`. At every *non-hot* slot the one-hot bit is 0, so its label equals the
//! garbler's mask and the evaluator recomputes `pad_i` itself. At the *hot* slot
//! the label differs, so `pad_hot` is the single pad it cannot derive. The `p_i`
//! pads are therefore an additive sharing the evaluator can open everywhere but
//! that one slot.
//!
//! **Delivering `a`.** The garbler sends one residue per affine map
//!
//! ```text
//!     diff = (Σ_i pad_i) + a            (mod p_i)
//! ```
//!
//! The evaluator subtracts every pad it knows: `pad_hot + a = diff − Σ_{i≠hot} pad_i`.
//! Now slot `hot` carries `pad_hot + a` and every other slot carries plain
//! `pad_i` — i.e. the secret `a` sits in slot `hot`, and all other slots are shares of 0.
//!
//! A linear combination weights slot `i` by `i` (and folds in
//! `b`), turning the `a` in slot `hot` into `a·(x mod p_i) + b`.
//! `(a, b)` stay hidden information-theoretically: single Z_p residues.

use crate::hash;
use crate::label::{LAMBDA, Label};

/// Cost of one IT-GC body batch (garbled-material footprint, for telemetry).
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchCost {
    /// Program bits the garbler emitted for the batch (the join diffs).
    pub program_bits: usize,
    /// NCF join width, in bits.
    pub join_complexity_ncf: usize,
    /// NCF CCRH block invocations.
    pub hash_count_ncf: usize,
}

/// Garbler-side output of one body batch, length `B` (one entry per member).
///
/// The evaluator knows `x` in cleartext (switch-private / data-public), so it
/// locates every switch's hot position itself; the garbler emits only the join
/// diffs.
#[derive(Debug)]
pub struct BodyBatchGarbleOutput {
    /// Join diff per batch member (NCF Z_p rep).
    pub join_diffs: Vec<u64>,
    /// Garbler's output mask per batch member (NCF Z_p rep).
    pub result_masks: Vec<u64>,
    /// Garbled-material footprint of this batch, for pipeline telemetry.
    pub cost: BatchCost,
}

/// Garbler kernel for one per-prime body batch (see the module docs): forms the
/// per-slot pads, sends `diff = Σ pad_i + a` per member.
///
/// `h_p_masks` are the carry-in CF Z_2 masks for the length-`p_i` one-hot;
/// `a_batch` / `b_batch` are the affine coefficients reduced mod `p_i`;
/// `truth_table[i] = i mod p_i` is the weight `g`. `group_id_base` offsets
/// the per-slot CCRH tweak.
pub fn body_batch_garble(
    p_i: u64,
    h_p_masks: &[Label],
    a_batch: &[u64],
    b_batch: &[u64],
    truth_table: &[u64],
    group_id_base: usize,
) -> BodyBatchGarbleOutput {
    let p = p_i as usize;
    let b = a_batch.len();
    assert_eq!(h_p_masks.len(), p, "h_p_masks length mismatch");
    assert_eq!(b_batch.len(), b, "a_batch and b_batch length mismatch");
    assert_eq!(truth_table.len(), p, "truth_table length mismatch");

    // pad_i = H(h_p[i]) for each one-hot slot, bulk-packed across the b members.
    let slot_hash = switch_hashes(h_p_masks, group_id_base, b, p_i);

    let mut join_diffs = Vec::with_capacity(b);
    let mut result_masks = Vec::with_capacity(b);
    for j in 0..b {
        let mut pad_sum = 0u64; // Σ_i pad_i
        let mut readout = neg_mod(b_batch[j], p_i); // seed −b so decoding restores +b
        for i in 0..p {
            let pad = member_pad(&slot_hash[i], j, p_i);
            pad_sum = mod_add(pad_sum, pad, p_i);
            readout = mod_add(readout, mod_mul(truth_table[i] % p_i, pad, p_i), p_i);
        }
        // The one residue sent per map: diff = Σ pad_i + a (mod_sub by −a adds a).
        join_diffs.push(mod_sub(pad_sum, neg_mod(a_batch[j], p_i), p_i));
        result_masks.push(readout);
    }

    BodyBatchGarbleOutput {
        join_diffs,
        result_masks,
        cost: batch_cost(p_i, b),
    }
}

/// Garbled-material footprint of a `b`-member body batch mod `p_i`: one join diff
/// of `lg|p_i|` bits per member, and `p_i` switch hashes of `b·lg|p_i|` bits each
/// (one CCRH block per `λ` bits).
fn batch_cost(p_i: u64, b: usize) -> BatchCost {
    let join_bits = b * hash::lg_modulus(p_i);
    BatchCost {
        program_bits: join_bits,
        join_complexity_ncf: join_bits,
        hash_count_ncf: p_i as usize * join_bits.div_ceil(LAMBDA),
    }
}

/// Evaluator kernel for one per-prime body batch: the label-side mirror of
/// [`body_batch_garble`] (see the module docs).
///
/// `hot = x mod p_i` is supplied in cleartext — the evaluator knows `x`, so the
/// control is never revealed. It recomputes every pad but `pad_hot`, opens the
/// hidden one from `diff`, and reads out the result. (`b_batch` is unused beyond
/// its length — `b`'s label is 0.) Returns the delivered residue per member; the
/// caller decodes `value = label − mask mod p_i`.
pub fn body_batch_eval(
    p_i: u64,
    hot: usize,
    h_p_labels: &[Label],
    join_diffs: &[u64],
    b_batch: &[u64],
    truth_table: &[u64],
    group_id_base: usize,
) -> Vec<u64> {
    let p = p_i as usize;
    let b = join_diffs.len();
    assert_eq!(h_p_labels.len(), p, "h_p_labels length mismatch");
    assert_eq!(b_batch.len(), b, "b_batch length mismatch");
    assert_eq!(truth_table.len(), p, "truth_table length mismatch");
    assert!(hot < p, "hot index out of range");

    // Same pads the garbler formed. At a non-hot slot the one-hot bit is 0, so
    // label == mask and the evaluator recomputes pad_i exactly; only pad_hot differs.
    let slot_hash = switch_hashes(h_p_labels, group_id_base, b, p_i);
    let g_hot = truth_table[hot] % p_i;

    (0..b)
        .map(|j| {
            let mut pad_sum = 0u64; // Σ_{i≠hot} pad_i  (every pad we can recompute)
            let mut readout = 0u64; // Σ_{i≠hot} g(i)·pad_i
            for i in 0..p {
                if i == hot {
                    continue;
                }
                let pad = member_pad(&slot_hash[i], j, p_i);
                pad_sum = mod_add(pad_sum, pad, p_i);
                readout = mod_add(readout, mod_mul(truth_table[i] % p_i, pad, p_i), p_i);
            }
            // Open the one hidden pad; the diff leaves `a` sitting on it:
            //   pad_hot + a = diff − Σ_{i≠hot} pad_i.
            let pad_hot = mod_sub(join_diffs[j], pad_sum, p_i);
            mod_add(readout, mod_mul(g_hot, pad_hot, p_i), p_i)
        })
        .collect()
}

/// The pad material `H(h_p[i])` for each one-hot slot, bulk-packed: each output
/// holds `b·lg|p_i|` pseudorandom bits, one `lg|p_i|`-bit pad per batch member.
/// Garbler and evaluator call this identically (on masks / labels); at every
/// non-hot slot the two agree, which is what lets `pad_i` line up.
fn switch_hashes(ohe: &[Label], group_id_base: usize, b: usize, p_i: u64) -> Vec<Vec<u8>> {
    let lg_p = hash::lg_modulus(p_i);
    ohe.iter()
        .enumerate()
        .map(|(i, l)| hash::hash_bulk(l, group_id_base + i, b * lg_p))
        .collect()
}

/// Member `j`'s pad `pad_i` (an NCF Z_p residue) sliced from slot `i`'s bulk hash.
#[inline(always)]
fn member_pad(slot_hash: &[u8], member_idx: usize, p: u64) -> u64 {
    match hash::extract_ncf(slot_hash, member_idx, p) {
        Label::Ncf(n) => n.rep,
        _ => unreachable!(),
    }
}

// NCF Z_p label algebra. A label is its residue, so the gate operations are
// plain modular arithmetic on `u64`.

/// Additive inverse mod p — the mask of a public constant c is −c.
#[inline(always)]
fn neg_mod(c: u64, p: u64) -> u64 {
    if c == 0 { 0 } else { p - c }
}

#[inline(always)]
fn mod_add(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

#[inline(always)]
fn mod_sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + p - b }
}

#[inline(always)]
fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    // `a, b < p`, so the product fits u64 iff `p ≤ 2^32`. At the concrete params
    // (primes ≤ 409) this holds with enormous margin; the assert pins it.
    debug_assert!(
        p <= (1u64 << 32),
        "NCF modulus {p} exceeds the u64 fast-path bound"
    );
    (a * b) % p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::CfLabel;
    use rand::{Rng, RngExt};

    fn rand_cf2_label<R: Rng>(rng: &mut R) -> Label {
        let coords: Vec<u64> = (0..crate::label::LAMBDA)
            .map(|_| rng.random_range(0..2u64))
            .collect();
        Label::Cf(CfLabel::from_coords(&coords, 2))
    }

    #[test]
    fn test_body_batch_kernel_round_trip_p3() {
        // Smallest case: p = 3, B = 4. h_p hot at index 1.
        let p_i = 3u64;
        let p = p_i as usize;
        let b = 4usize;
        let mut rng = rand::rng();

        // Construct h_p as the OHE of value 1 (index 1 hot).
        let hot_idx = 1usize;
        let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
        let h_p_labels: Vec<Label> = (0..p)
            .map(|i| {
                if i == hot_idx {
                    // h_p[hot] = 1 ⇒ label = mask + Δ_2 (we just need the LSB to differ).
                    let mut coords = h_p_masks[i].as_cf().to_coords();
                    coords[0] ^= 1;
                    Label::Cf(CfLabel::from_coords(&coords, 2))
                } else {
                    h_p_masks[i].clone()
                }
            })
            .collect();

        let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let truth: Vec<u64> = (0..p_i).collect();

        let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &truth, 0);
        let result_labels = body_batch_eval(
            p_i,
            hot_idx,
            &h_p_labels,
            &g_out.join_diffs,
            &b_batch,
            &truth,
            0,
        );

        // result_j.label - result_j.mask = result_j.value = b_j + hot · a_j (mod p_i).
        for j in 0..b {
            let value = mod_sub(result_labels[j], g_out.result_masks[j], p_i);
            let expected = (b_batch[j] + (hot_idx as u64) * a_batch[j]) % p_i;
            assert_eq!(
                value, expected,
                "j={j}, hot={hot_idx}, a={}, b={}, p={p_i}",
                a_batch[j], b_batch[j],
            );
        }

        // Accidentally getting the hash invocations wrong would fail this — just keep h_p_masks alive.
        let _ = h_p_masks;
    }

    #[test]
    fn test_body_batch_kernel_all_hot_positions_p7() {
        // Sweep every possible hot index, mid-size prime.
        let p_i = 7u64;
        let p = p_i as usize;
        let b = 6usize;
        let mut rng = rand::rng();
        let truth: Vec<u64> = (0..p_i).collect();

        for hot_idx in 0..p {
            let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
            let h_p_labels: Vec<Label> = (0..p)
                .map(|i| {
                    if i == hot_idx {
                        let mut coords = h_p_masks[i].as_cf().to_coords();
                        coords[0] ^= 1;
                        Label::Cf(CfLabel::from_coords(&coords, 2))
                    } else {
                        h_p_masks[i].clone()
                    }
                })
                .collect();

            let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
            let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();

            let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &truth, 0);
            let result_labels = body_batch_eval(
                p_i,
                hot_idx,
                &h_p_labels,
                &g_out.join_diffs,
                &b_batch,
                &truth,
                0,
            );

            for j in 0..b {
                let value = mod_sub(result_labels[j], g_out.result_masks[j], p_i);
                let expected = (b_batch[j] + (hot_idx as u64) * a_batch[j]) % p_i;
                assert_eq!(
                    value, expected,
                    "hot_idx={hot_idx}, j={j}, a={}, b={}",
                    a_batch[j], b_batch[j],
                );
            }
        }
    }
}
