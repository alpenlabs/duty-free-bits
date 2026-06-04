//! Information-theoretic GC: the per-prime body batch.
//!
//! This is Phase 3 of the construction (paper §6.3): given the computational
//! GC's CF binary one-hot `h_p` of `x mod p_i`, it delivers `a·x + b mod p_i`
//! using `H(h_p[k])` as one-time pads. The masking of the secret `(a, b)` is
//! information-theoretic — single NCF Z_p residues, no λ blowup — which is why
//! it is implemented directly here rather than through the general switch engine.
//!
//! A body batch is one `hot_to_ring_bulk(h_p, identity, a, b)` from
//! [`crate::components::convert`], which expands into `ohe_scale_bulk` followed
//! by a ring readout. Written as the protocol's gates, for OHE position `i` and
//! batch member `j`:
//!
//!   * **switch** `o_{i,j} = H(h_p[i]) + z`, with the data input `z = 0`. This
//!     is the scaled OHE share: it equals `a_j` at the hot position and `0`
//!     everywhere else.
//!   * **join** of `acc_j = Σ_i o_{i,j}` against `a_j`. This ties the hot share
//!     to `a_j`; on the evaluator side it is what recovers that share.
//!   * **readout** `result_j = Σ_i g(i)·o_{i,j} + b_j`, with `g = identity`.
//!
//! Routing the 800 body batches at production scale through a full `System` (a
//! worklist plus a per-wire `Vec<Option<Label>>`) is wasteful: the live state is
//! just the carry-in OHE and a few per-batch accumulators. This module runs
//! exactly the three gates above, inlined on NCF Z_p residues. There a label is
//! a single `u64`, so the gate algebra (`add`/`sub`/`scalar_mul`) is plain
//! modular `u64` arithmetic (`mod_add`/`mod_sub`/`mod_mul`).
//!
//! The two-party split is preserved. [`body_batch_garble`] runs the gates on the
//! garbler's masks and emits the program (the join diffs); [`body_batch_eval`]
//! runs them on the evaluator's labels and consumes it. The two can run in
//! separate processes.

use crate::garble::hash;
use crate::garble::{BatchCost, Label};
use crate::system::LAMBDA_BITS;

/// Garbler-side output of one body batch, length `B` (one entry per member).
///
/// The evaluator knows `x` in cleartext (switch-private / data-public), so it
/// can locate every switch's hot position itself. We therefore skip the usual
/// point-and-permute ctrl-LSB reveal and emit only the join diffs.
#[derive(Debug)]
pub struct BodyBatchGarbleOutput {
    /// Join diff per batch member (NCF Z_p rep).
    pub join_diffs: Vec<u64>,
    /// Garbler's output mask per batch member (NCF Z_p rep).
    pub result_masks: Vec<u64>,
    /// Garbled-material footprint of this batch, for pipeline telemetry.
    pub cost: BatchCost,
}

/// Garbler kernel for one per-prime body batch: runs the switch / join / readout
/// gates on the garbler's masks (see the module docs).
///
/// `h_p_masks` are the carry-in CF Z_2 masks for the length-`p_i` OHE;
/// `a_batch` / `b_batch` are the affine coefficients reduced mod `p_i`;
/// `truth_table[i] = i mod p_i` is `g`. `group_id_base` offsets the per-position
/// CCRH tweak and must match the value passed to [`body_batch_eval`].
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

    // H(h_p[i]) for each OHE position, the switch hash shared by all b members.
    let switch_hash = switch_hashes(h_p_masks, group_id_base, b, p_i);

    let mut join_diffs = Vec::with_capacity(b);
    let mut result_masks = Vec::with_capacity(b);
    for j in 0..b {
        // The constants z and b_j contribute their masks: z.mask = 0 (so the
        // switch output o_{i,j}.mask is just the hash), and b_j.mask = −b_j.
        let mut acc = 0u64; // acc_j.mask = Σ_i o_{i,j}.mask
        let mut result = neg_mod(b_batch[j], p_i); // readout, seeded with b_j.mask
        for i in 0..p {
            let o = ncf_slice(&switch_hash[i], j, p_i); // switch: o_{i,j}.mask
            acc = mod_add(acc, o, p_i);
            result = mod_add(result, mod_mul(truth_table[i] % p_i, o, p_i), p_i);
        }
        // join(acc_j, a_j) reveals diff = acc_j.mask − a_j.mask  (a_j.mask = −a_j).
        join_diffs.push(mod_sub(acc, neg_mod(a_batch[j], p_i), p_i));
        result_masks.push(result);
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
        hash_count_ncf: p_i as usize * join_bits.div_ceil(LAMBDA_BITS),
    }
}

/// Evaluator kernel for one per-prime body batch: the label-side view of the
/// same gates [`body_batch_garble`] runs on masks.
///
/// `hot = x mod p_i` is supplied directly: the evaluator knows `x` in cleartext,
/// so there is no point-and-permute reconstruction. `b_batch` is present for
/// symmetry — b_j is a public constant whose label is 0, so its values are
/// unused. Returns one label per batch member.
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

    // The same H(h_p[i]) the garbler used. At a non-hot position the OHE entry is
    // 0, so the evaluator's label equals the garbler's mask and the switch output
    // o_{i,j}.label matches o_{i,j}.mask — i.e. we can recompute it directly.
    let switch_hash = switch_hashes(h_p_labels, group_id_base, b, p_i);
    let g_hot = truth_table[hot] % p_i;

    (0..b)
        .map(|j| {
            // Accumulate the non-hot switch outputs. The readout starts at 0
            // because the constant b_j's label is 0.
            let mut acc = 0u64; // Σ_{i≠hot} o_{i,j}.label
            let mut result = 0u64; // Σ_{i≠hot} g(i)·o_{i,j}.label
            for i in 0..p {
                if i == hot {
                    continue;
                }
                let o = ncf_slice(&switch_hash[i], j, p_i);
                acc = mod_add(acc, o, p_i);
                result = mod_add(result, mod_mul(truth_table[i] % p_i, o, p_i), p_i);
            }
            // At the hot position the label differs from the mask, so o_{hot,j}
            // can't be hashed. The join recovers it: by acc_j.label = a_j.label,
            // o_{hot,j}.label = diff − Σ_{i≠hot} o_{i,j}.label.
            let o_hot = mod_sub(join_diffs[j], acc, p_i);
            mod_add(result, mod_mul(g_hot, o_hot, p_i), p_i)
        })
        .collect()
}

/// The switch hash `H(h_p[i])` for each OHE position, bulk-packed: each output
/// holds `b·lg|p_i|` pseudorandom bits, one `lg|p_i|`-bit slice per batch member.
/// Garbler and evaluator call this identically (on masks / labels); at every
/// non-hot position the two agree, which is what lets the gates line up.
fn switch_hashes(ohe: &[Label], group_id_base: usize, b: usize, p_i: u64) -> Vec<Vec<u8>> {
    let lg_p = hash::lg_modulus(p_i);
    ohe.iter()
        .enumerate()
        .map(|(i, l)| hash::hash_bulk(l, group_id_base + i, b * lg_p))
        .collect()
}

/// One member's switch output `o_{i,j}` (an NCF Z_p residue) from a position's
/// bulk switch hash.
#[inline(always)]
fn ncf_slice(switch_hash: &[u8], member_idx: usize, p: u64) -> u64 {
    match hash::extract_ncf(switch_hash, member_idx, p) {
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
    (a * b) % p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garble::label::CfLabel;
    use rand::Rng;

    fn rand_cf2_label<R: Rng>(rng: &mut R) -> Label {
        let coords: Vec<u64> = (0..crate::garble::label::LAMBDA)
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

    /// Security regression: each body switch's pad is `H(h_p[i], group_id_base+i)`
    /// (paper §1.3, `ct = Σ_i H(L_i^0, nonce) + a`). The pad hides `a_j` only if
    /// the nonce is fresh. Two batches of a prime share the same `h_p` seed, so
    /// they MUST use disjoint nonce bases; otherwise the shared pad cancels and
    /// `join_diff^0_j − join_diff^1_j` leaks `a^0_j − a^1_j`. This pins both the
    /// leak (under a reused base) and the fix (under distinct bases).
    #[test]
    fn test_distinct_nonces_prevent_two_time_pad_leak() {
        let p_i = 31u64;
        let p = p_i as usize;
        let b = 6usize;
        let mut rng = rand::rng();
        let truth: Vec<u64> = (0..p_i).collect();

        // Same OHE (seed) for both batches, as in a single prime's body.
        let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
        let a0: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let a1: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let zeros = vec![0u64; b];

        // Reused nonce base (the bug): the pads cancel exactly, leaking a0 − a1
        // in EVERY position. This also confirms the test can detect the bug.
        let bug0 = body_batch_garble(p_i, &h_p_masks, &a0, &zeros, &truth, 0);
        let bug1 = body_batch_garble(p_i, &h_p_masks, &a1, &zeros, &truth, 0);
        for j in 0..b {
            let leak = mod_sub(bug0.join_diffs[j], bug1.join_diffs[j], p_i);
            let secret = mod_sub(a0[j], a1[j], p_i);
            assert_eq!(leak, secret, "reused nonce must leak a0−a1 at j={j}");
        }

        // Distinct nonce bases (the fix): the pads differ, so the simple leak
        // relation is broken. (Spacing must be ≥ p so position nonces don't
        // overlap; the streaming driver advances the base by p_i per batch.)
        let fix0 = body_batch_garble(p_i, &h_p_masks, &a0, &zeros, &truth, 0);
        let fix1 = body_batch_garble(p_i, &h_p_masks, &a1, &zeros, &truth, p);
        let leaked_positions = (0..b)
            .filter(|&j| {
                let leak = mod_sub(fix0.join_diffs[j], fix1.join_diffs[j], p_i);
                let secret = mod_sub(a0[j], a1[j], p_i);
                leak == secret
            })
            .count();
        // With fresh pads the relation holds only by chance (~1/p per position);
        // it must not hold across the board the way the reused base forces.
        assert!(
            leaked_positions < b,
            "distinct nonces must break the two-time-pad leak (leaked {leaked_positions}/{b})"
        );
    }

    /// Generalize the leak check to the real streaming pattern: K consecutive
    /// batches of one prime (same `h_p` seed) with the driver's nonce scheme
    /// (`base += p_i` per batch). No pair of batches may share a pad, so the
    /// two-time-pad relation must be broken for every pair.
    #[test]
    fn test_streaming_nonce_scheme_no_pad_reuse() {
        let p_i = 31u64;
        let p = p_i as usize;
        let b = 5usize;
        let k = 4usize;
        let mut rng = rand::rng();
        let truth: Vec<u64> = (0..p_i).collect();
        let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
        let zeros = vec![0u64; b];

        let mut batches: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();
        for batch in 0..k {
            let a: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
            let base = batch * p; // matches affine.rs `group_id_base += p_i`
            let g = body_batch_garble(p_i, &h_p_masks, &a, &zeros, &truth, base);
            batches.push((a, g.join_diffs));
        }
        for u in 0..k {
            for v in (u + 1)..k {
                let leaked = (0..b)
                    .filter(|&j| {
                        mod_sub(batches[u].1[j], batches[v].1[j], p_i)
                            == mod_sub(batches[u].0[j], batches[v].0[j], p_i)
                    })
                    .count();
                assert!(
                    leaked < b,
                    "pad reuse between batch {u} and {v} (leaked {leaked}/{b})"
                );
            }
        }
    }
}
