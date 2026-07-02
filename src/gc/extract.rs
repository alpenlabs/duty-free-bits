//! Step 2 / 4 (computational GC): decompose a ring word into its sub-chunk
//! bits + a binary one-hot + the peel upcast — the fused `word_to_bin_up`.
//!
//! Per prime: form `r_i = Σ_c coeff_c·w_c`, split it into ≤8-bit sub-chunks,
//! and for each run the [`super::onehot`] tree + casts. The width-`l` casts do
//! double duty — their weighted class sums both peel the bits (LSB-first) and
//! give the upcast `Σ p·A_p` that reconstructs the sub-chunk value for the next
//! peel. Outputs the first sub-chunk's binary one-hot + the remaining bits,
//! which [`super::fold`] consumes.

use super::onehot::*;
use crate::hash;
use crate::label::{LAMBDA, Label, delta_r};


/// Sub-chunk widths for decomposing an `ell`-bit value, each at most
/// `max_width` bits (greedy: `ell = 22, max = 8 → [8, 8, 6]`).
pub fn compute_sub_widths(ell: u32, max_width: u32) -> Vec<u32> {
    assert!(ell > 0 && max_width > 0);
    let mut widths = vec![];
    let mut remaining = ell;
    while remaining > 0 {
        let w = remaining.min(max_width);
        widths.push(w);
        remaining -= w;
    }
    widths
}

/// The extract step's lane machinery represents width-`l` wires as u32
/// lanes, so every sub-chunk width must be in `2..=31`; a width-1 trailing
/// sub-chunk (`ell ≡ 1 mod 8` under `compute_sub_widths(·, 8)`) is not
/// supported. Rejected loudly in release builds too.
fn assert_sub_widths(sub_widths: &[u32]) {
    assert!(
        sub_widths.iter().all(|&w| (2..=31).contains(&w)),
        "extract step requires sub-chunk widths in 2..=31 (got {sub_widths:?}); \
         width-1 trailing sub-chunks are not supported"
    );
}

/// (bulk ids, solo ids) consumed by one extract step.
pub fn extract_nonces(sub_widths: &[u32]) -> (u64, u64) {
    let mut bulk = 0;
    let mut solo = 0;
    for &w in sub_widths {
        bulk += tree_ids(w);
        solo += 1u64 << w;
    }
    (bulk, solo)
}

/// Garbler-side output of one extract step.
#[derive(Debug)]
pub struct ExtractGarbleOutput {
    /// Masks of the first sub-chunk's binary one-hot (CF Z₂, 2^{w₀} entries).
    pub first_bin_hot_masks: Vec<Label>,
    /// Masks of the fold bits (sub-chunks 1.., LSB-first within each).
    pub fold_bit_masks: Vec<Label>,
    /// Per sub-chunk: the k−1 scale-join diffs and the width-`l` pin diff.
    pub diffs: Vec<SubChunkDiffs>,
    /// Ledger footprint.
    pub cost: StepCost,
}

/// The communicated material of one sub-chunk stage.
#[derive(Debug)]
pub struct SubChunkDiffs {
    /// Scale-join diff per tree level (k−1 of them, Z₂).
    pub scale: Vec<Z2>,
    /// Pin-join diff (width `l` = the stage's cast width).
    pub pin: Wide,
}

/// Per-sub-chunk state shared by garble/eval: widths and nonce bases.
struct StagePlan {
    k: u32,
    l: u32,
    last: bool,
    bulk_base: u64,
    solo_base: u64,
}

fn plan_stages(ell: u32, sub_widths: &[u32], bulk_base: u64, solo_base: u64) -> Vec<StagePlan> {
    let mut rem_bits = ell;
    let mut bulk = bulk_base;
    let mut solo = solo_base;
    let mut plans = Vec::with_capacity(sub_widths.len());
    for (q, &k) in sub_widths.iter().enumerate() {
        let last = q == sub_widths.len() - 1;
        let l = if last { k } else { rem_bits };
        plans.push(StagePlan {
            k,
            l,
            last,
            bulk_base: bulk,
            solo_base: solo,
        });
        bulk += tree_ids(k);
        solo += 1u64 << k;
        rem_bits -= k;
    }
    plans
}

/// Garbler side: `r = Σ_c coeff_c·w_c`, then the fused `word_to_bin_up`
/// per sub-chunk. X-blind: hashes every tree slot and every cast, including
/// the (unknown) hot ones.
///
/// `chunk_word_masks` are CF Z_{2^ell} masks of the chunk words; `nonce_bulk`
/// / `nonce_solo` are fresh windows sized by [`extract_nonces`].
pub fn extract_batch_garble(
    chunk_word_masks: &[Label],
    coeffs: &[u64],
    sub_widths: &[u32],
    delta: u128,
    nonce_bulk: u64,
    nonce_solo: u64,
) -> ExtractGarbleOutput {
    assert_sub_widths(sub_widths);
    let ell = chunk_word_masks[0].k();
    assert_eq!(
        sub_widths.iter().sum::<u32>(),
        ell,
        "sub-chunk widths must sum to the chunk-word width"
    );
    debug_assert!(chunk_word_masks.iter().all(|m| m.k() == ell));
    let mask_ell = ((1u64 << ell) - 1) as u32;
    let d2 = {
        let d = delta_r(delta, 2);
        let raw = d.raw_bits();
        [raw[0], raw[1]]
    };

    // r mask = Σ coeff_c · X_{w_c} (free lane arithmetic).
    let mut rem = [0u32; LAMBDA];
    for (m, &c) in chunk_word_masks.iter().zip(coeffs) {
        wide_madd(&mut rem, (c & mask_ell as u64) as u32, &wide_of(m));
    }

    let plans = plan_stages(ell, sub_widths, nonce_bulk, nonce_solo);
    let mut first_bin_hot_masks = Vec::new();
    let mut fold_bit_masks = Vec::new();
    let mut diffs = Vec::with_capacity(plans.len());
    let mut cost = StepCost::default();
    let mut rem_bits = ell;

    for (q, p) in plans.iter().enumerate() {
        let (k, l) = (p.k, p.l);
        let mask_rem = ((1u64 << rem_bits) - 1) as u32;

        // sub = rem mod 2^k as width-k lanes; bit 0 is native.
        let mask_k = ((1u64 << k) - 1) as u32;
        let mut sub = [0u32; LAMBDA];
        for (s, r) in sub.iter_mut().zip(&rem) {
            *s = r & mask_k;
        }
        let bit0 = pack_lane_bit(&sub, 0);

        // Doubling tree, level-major over every slot.
        let lvl1 = [[bit0[0] ^ d2[0], bit0[1] ^ d2[1]], bit0];
        let (leaves, ysums) = tree_garble(k, lvl1, p.bulk_base);

        // Width-l casts of every leaf.
        let n = 1usize << k;
        let mut casts = vec![[0u32; LAMBDA]; n];
        for (pp, cast) in casts.iter_mut().enumerate() {
            hash_cast(&leaves[pp], p.solo_base + pp as u64, l, cast);
        }

        // Residue bank + root; bits i = 1..k−1 peel from res_i.
        let (res_chain, root) = residues_all(casts);
        let mut bit_masks: Vec<Z2> = Vec::with_capacity(k as usize);
        bit_masks.push(bit0);
        for i in 1..k {
            bit_masks.push(peel_bit(&sub, &res_chain[i as usize - 1], i, k));
        }

        // Scale diffs (⊕ y_m ⊕ X_{bit_m}) and the pin diff (root + Δ_l:
        // diff = X_root − X_one with X_one = −Δ_l).
        let scale: Vec<Z2> = ysums
            .iter()
            .zip(bit_masks.iter().skip(1))
            .map(|(y, b)| [y[0] ^ b[0], y[1] ^ b[1]])
            .collect();
        let mut pin = root;
        let dl = delta_r(delta, 1u64 << l);
        wide_add(&mut pin, &wide_of(&dl));
        let mask_l = ((1u64 << l) - 1) as u32;
        for lane in pin.iter_mut() {
            *lane &= mask_l;
        }

        if q == 0 {
            first_bin_hot_masks = leaves.iter().map(|&w| z2_label(w)).collect();
        } else {
            fold_bit_masks.extend(bit_masks.iter().map(|&w| z2_label(w)));
        }

        // Peel: rem = (rem − up) >> k; up = res_k (the level-k functional).
        if !p.last {
            let up = &res_chain[k as usize - 1];
            let mut d = rem;
            wide_sub(&mut d, up);
            for lane in d.iter_mut() {
                *lane = (*lane & mask_rem) >> k;
            }
            rem = d;
            rem_bits -= k;
        }

        diffs.push(SubChunkDiffs { scale, pin });
        cost.add(stage_cost(k, l));
    }

    ExtractGarbleOutput {
        first_bin_hot_masks,
        fold_bit_masks,
        diffs,
        cost,
    }
}

/// Evaluator side: the label-side mirror of [`extract_batch_garble`],
/// following the straight-line `word_to_bin_up` schedule. `r_value` is the
/// evaluator's cleartext `Σ coeff_c·v_c` (it knows `x`): every hot index and
/// class residue is index arithmetic; hot slots are solved through the scale
/// joins, and the hot cast class through the pin join.
pub fn extract_batch_eval(
    chunk_word_labels: &[Label],
    coeffs: &[u64],
    r_value: u64,
    sub_widths: &[u32],
    diffs: &[SubChunkDiffs],
    nonce_bulk: u64,
    nonce_solo: u64,
) -> (Vec<Label>, Vec<Label>) {
    assert_sub_widths(sub_widths);
    let ell = chunk_word_labels[0].k();
    assert_eq!(
        sub_widths.iter().sum::<u32>(),
        ell,
        "sub-chunk widths must sum to the chunk-word width"
    );
    assert_eq!(diffs.len(), sub_widths.len(), "one SubChunkDiffs per sub-chunk");
    debug_assert!(chunk_word_labels.iter().all(|m| m.k() == ell));
    let mask_ell = ((1u64 << ell) - 1) as u32;

    let mut rem = [0u32; LAMBDA];
    for (m, &c) in chunk_word_labels.iter().zip(coeffs) {
        wide_madd(&mut rem, (c & mask_ell as u64) as u32, &wide_of(m));
    }
    let mut rem_value = r_value;
    let mut rem_bits = ell;

    let plans = plan_stages(ell, sub_widths, nonce_bulk, nonce_solo);
    let mut first_bin_hot_labels = Vec::new();
    let mut fold_bit_labels = Vec::new();

    for (q, (p, dd)) in plans.iter().zip(diffs).enumerate() {
        let k = p.k;
        let n = 1usize << k;
        let mask_rem = ((1u64 << rem_bits) - 1) as u32;
        let mask_k = ((1u64 << k) - 1) as u32;
        let s = rem_value & mask_k as u64; // this stage's cleartext sub-chunk

        let mut sub = [0u32; LAMBDA];
        for (dst, r) in sub.iter_mut().zip(&rem) {
            *dst = r & mask_k;
        }
        let bit0 = pack_lane_bit(&sub, 0);

        // The hold-all tree store (2^{m} slots per level) plus per-level y-sums.
        let mut lvl: Vec<Vec<Z2>> = (0..=k).map(|m| vec![[0u64; 2]; 1usize << m]).collect();
        // NOT is affine: both level-1 slots carry the bit-0 label.
        lvl[1][0] = bit0;
        lvl[1][1] = bit0;
        let mut ysum = vec![[0u64; 2]; k as usize];

        // Accumulator bank over solved casts: R_j = Σ (p mod 2^j)·A_p,
        // T_j = Σ A_p for j = 1..=k (width l).
        let mut acc_r = vec![[0u32; LAMBDA]; k as usize];
        let mut acc_t = vec![[0u32; LAMBDA]; k as usize];
        // L_root from the pin join: the constant-1 wire's label is 0.
        let root = dd.pin;

        // Scratch for the per-class subtree fold (max class size 2^{k−1}).
        let mut sums: Vec<Wide> = Vec::with_capacity(1usize << (k - 1));

        // Round 0: bit 0 is native; expand its complement class.
        let mut bit_labels: Vec<Z2> = Vec::with_capacity(k as usize);
        bit_labels.push(bit0);
        expand_and_fold_class(
            p,
            (s & 1) ^ 1,
            1,
            &mut lvl,
            &mut ysum,
            &mut acc_r,
            &mut acc_t,
            &mut sums,
        );

        // Rounds 1..k−1: residue → bit i → hot solve → expand the new class.
        for i in 1..k {
            let qi = s & ((1u64 << i) - 1);
            // res_i = R_i + q·(L_root − T_i): the hot class enters via the pin.
            let ii = i as usize - 1;
            let mut res = root;
            wide_sub(&mut res, &acc_t[ii]);
            let mut res_full = acc_r[ii];
            wide_madd(&mut res_full, qi as u32, &res);
            let bit_i = peel_bit(&sub, &res_full, i, k);
            bit_labels.push(bit_i);

            // Hot slot of level i solves through the scale join.
            let d = dd.scale[ii];
            let ys = ysum[i as usize];
            let y_hot = [
                bit_i[0] ^ d[0] ^ ys[0],
                bit_i[1] ^ d[1] ^ ys[1],
            ];
            let hot = qi as usize;
            let parent = lvl[i as usize][hot];
            lvl[i as usize + 1][hot] = [parent[0] ^ y_hot[0], parent[1] ^ y_hot[1]];
            lvl[i as usize + 1][hot + (1usize << i)] = y_hot;

            let b = (s >> i) & 1;
            expand_and_fold_class(
                p,
                qi | ((b ^ 1) << i),
                i + 1,
                &mut lvl,
                &mut ysum,
                &mut acc_r,
                &mut acc_t,
                &mut sums,
            );
        }

        if q == 0 {
            first_bin_hot_labels = lvl[k as usize].iter().map(|&w| z2_label(w)).collect();
            debug_assert_eq!(first_bin_hot_labels.len(), n);
        } else {
            fold_bit_labels.extend(bit_labels.iter().map(|&w| z2_label(w)));
        }

        // Peel: up = R_k + s·(L_root − T_k), rem = (rem − up) >> k.
        if !p.last {
            let ki = k as usize - 1;
            let mut up = acc_r[ki];
            let mut hot_part = root;
            wide_sub(&mut hot_part, &acc_t[ki]);
            wide_madd(&mut up, s as u32, &hot_part);
            let mut d = rem;
            wide_sub(&mut d, &up);
            for lane in d.iter_mut() {
                *lane = (*lane & mask_rem) >> k;
            }
            rem = d;
            rem_value >>= k;
            rem_bits -= k;
        }
    }

    (first_bin_hot_labels, fold_bit_labels)
}

/// Bulk-id base of level `m` within a stage (slots of levels 1..m precede it).
#[inline]
fn nonce_bulk_stage(p: &StagePlan, m: u32) -> u64 {
    p.bulk_base + ((1u64 << m) - 2)
}

/// Expand the newly-zero slot `z` (level `lz`) to the leaves — every switch
/// en route is closed — then cast its leaves and fold the class into
/// accumulator levels lz..=k.
///
/// The fold is subtree-structured: leaves are `p_c = z + c·2^lz`, so
/// `p_c mod 2^j = z + (c mod 2^{j−lz})·2^lz`, and the class's contribution to
/// `R_j` collapses to `z·T + 2^lz·W_{j−lz}` with `T = Σ_c A_c` and
/// `W_t = Σ_c (c mod 2^t)·A_c` — the W chain comes from the same
/// halving-plus-H recurrence as [`residues_all`], costing O(2^{k−lz}) lane
/// ops per class instead of O((k−lz)·2^{k−lz}) per-leaf folds.
#[allow(clippy::too_many_arguments)]
fn expand_and_fold_class(
    p: &StagePlan,
    z: u64,
    lz: u32,
    lvl: &mut [Vec<Z2>],
    ysum: &mut [Z2],
    acc_r: &mut [Wide],
    acc_t: &mut [Wide],
    sums: &mut Vec<Wide>,
) {
    let (k, l) = (p.k, p.l);
    for m in lz..k {
        let base = nonce_bulk_stage(p, m);
        for c in 0..(1u64 << (m - lz)) {
            let j = (z + (c << lz)) as usize;
            let bj = lvl[m as usize][j];
            let y = hash::hash_z2(&bj, base + j as u64);
            ysum[m as usize][0] ^= y[0];
            ysum[m as usize][1] ^= y[1];
            lvl[m as usize + 1][j] = [bj[0] ^ y[0], bj[1] ^ y[1]];
            lvl[m as usize + 1][j + (1usize << m)] = y;
        }
    }

    let tmax = (k - lz) as usize;
    sums.clear();
    for c in 0..(1u64 << tmax) {
        let pp = z + (c << lz);
        let mut cast = [0u32; LAMBDA];
        hash_cast(&lvl[k as usize][pp as usize], p.solo_base + pp, l, &mut cast);
        sums.push(cast);
    }
    // Halve to the class total, gathering the high-half sums H_t.
    let mut highs: Vec<Wide> = Vec::with_capacity(tmax);
    while sums.len() > 1 {
        let mid = sums.len() / 2;
        let mut high = [0u32; LAMBDA];
        let (lo, hi) = sums.split_at_mut(mid);
        for d in 0..mid {
            wide_add(&mut high, &hi[d]);
            wide_add(&mut lo[d], &hi[d]);
        }
        sums.truncate(mid);
        highs.push(high);
    }
    let total = sums[0];
    // W chain: W_0 = 0, W_{t+1} = W_t + 2^t·H_t (H_t in reverse gather order).
    let mut w_chain: Vec<Wide> = Vec::with_capacity(tmax + 1);
    w_chain.push([0u32; LAMBDA]);
    for (t, high) in highs.iter().rev().enumerate() {
        let mut w = w_chain[t];
        wide_madd(&mut w, 1u32 << t, high);
        w_chain.push(w);
    }
    // ΔR_j = z·T + 2^lz·W_{j−lz}, ΔT_j = T, for j = lz..=k.
    for j in lz..=k {
        let ji = j as usize - 1;
        wide_madd(&mut acc_r[ji], z as u32, &total);
        wide_madd(&mut acc_r[ji], 1u32 << lz, &w_chain[(j - lz) as usize]);
        wide_add(&mut acc_t[ji], &total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{self, LAMBDA};
    use rand::Rng;

    fn rand_wide(rng: &mut impl Rng, modulus: u64) -> Label {
        let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
        Label::from_coords(&coords, modulus)
    }

    #[test]
    fn test_compute_sub_widths() {
        assert_eq!(compute_sub_widths(22, 8), vec![8, 8, 6]);
        assert_eq!(compute_sub_widths(16, 8), vec![8, 8]);
        assert_eq!(compute_sub_widths(3, 8), vec![3]);
        assert_eq!(compute_sub_widths(4, 1), vec![1, 1, 1, 1]);
    }

    /// The extract step rejects width-1 sub-chunks loudly (the lane
    /// machinery requires widths ≥ 2).
    #[test]
    #[should_panic(expected = "sub-chunk widths in 2..=31")]
    fn test_extract_rejects_width1() {
        let mut rng = rand::rng();
        let delta: u128 = rng.random::<u128>() | 1;
        let masks = vec![rand_wide(&mut rng, 1 << 9)];
        let _ = extract_batch_garble(&masks, &[1], &[8, 1], delta, 0, 0);
    }

    /// Extract step: OHE labels + fold-bit labels satisfy the carry
    /// invariant for every r, at the production shape and a small one.
    #[test]
    fn test_extract_label_mask_invariant() {
        let mut rng = rand::rng();
        let delta: u128 = rng.random::<u128>() | 1;
        let d2 = delta_r(delta, 2);
    
        for (ell, sub_widths, coeffs) in [
            (8u32, vec![4u32, 4], vec![1u64, 16]),
            (22, vec![8, 8, 6], vec![1, 37, 4123]),
        ] {
            let m_ell = 1u64 << ell;
            let word_masks: Vec<Label> =
                (0..coeffs.len()).map(|_| rand_wide(&mut rng, m_ell)).collect();
            let (bulk_n, solo_n) = extract_nonces(&sub_widths);
    
            let g = extract_batch_garble(&word_masks, &coeffs, &sub_widths, delta, 100, 200);
            // Emitted material must be canonical: every pin lane < 2^l, where
            // l is the stage's cast width (the remainder width, or k on the
            // last stage).
            let mut rb = ell;
            for (q, (dd, &k)) in g.diffs.iter().zip(&sub_widths).enumerate() {
                let l = if q == sub_widths.len() - 1 { k } else { rb };
                rb -= k;
                let lim = 1u32 << l;
                assert!(dd.pin.iter().all(|&lane| lane < lim), "pin lane not canonical");
            }
            let expect_cost: usize = {
                let mut rb = ell;
                sub_widths
                    .iter()
                    .enumerate()
                    .map(|(q, &k)| {
                        let l = if q == sub_widths.len() - 1 { k } else { rb };
                        rb -= k;
                        (1usize << k) - 2 + l as usize * (1usize << k)
                    })
                    .sum()
            };
            assert_eq!(g.cost.hash_count_cf, expect_cost);
            let _ = (bulk_n, solo_n);
    
            // Random chunk-word values; r = Σ coeff·v mod 2^ell.
            for _ in 0..24 {
                let values: Vec<u64> =
                    (0..coeffs.len()).map(|_| rng.random_range(0..m_ell)).collect();
                let r: u64 = values
                    .iter()
                    .zip(&coeffs)
                    .fold(0u64, |a, (&v, &c)| (a + v.wrapping_mul(c)) & (m_ell - 1));
                let dl_ell = delta_r(delta, m_ell);
                let word_labels: Vec<Label> = word_masks
                    .iter()
                    .zip(&values)
                    .map(|(m, &v)| label::add(m, &label::scalar_mul(v, &dl_ell)))
                    .collect();
    
                let (fbh, fbits) = extract_batch_eval(
                    &word_labels,
                    &coeffs,
                    r,
                    &sub_widths,
                    &g.diffs,
                    100,
                    200,
                );
    
                // first_bin_hot: label = mask + [p == r mod 2^{w0}]·Δ₂.
                let w0 = sub_widths[0];
                let low = (r & ((1u64 << w0) - 1)) as usize;
                for (p, (l, m)) in fbh.iter().zip(&g.first_bin_hot_masks).enumerate() {
                    let expect = if p == low {
                        label::add(m, &d2)
                    } else {
                        m.clone()
                    };
                    assert_eq!(*l, expect, "fbh[{p}] r={r} ell={ell}");
                }
                // Fold bits: bit j of sub-chunk q.
                let mut pos = w0;
                for (l, m) in fbits.iter().zip(&g.fold_bit_masks) {
                    let expect = if (r >> pos) & 1 == 1 {
                        label::add(m, &d2)
                    } else {
                        m.clone()
                    };
                    assert_eq!(*l, expect, "fold bit at {pos}, r={r} ell={ell}");
                    pos += 1;
                }
                assert_eq!(pos, ell);
            }
        }
    }
}

#[cfg(test)]
mod micro {
    use super::*;
    use std::time::Instant;

    /// Attribute extract-step time to its primitives at production shape.
    /// Run: cargo test --release --lib bench_extract_micro -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_extract_micro() {
        let reps = 80 * 30; // 80 primes × 30 reps, i.e. "per bench_axb_stages run"
        let ctrl: Z2 = [0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321];
        let mut cast = [7u32; LAMBDA];

        // 1. Cast hashing: (256 @ l=22) + (256 @ l=14) + (64 @ l=6).
        let t = Instant::now();
        for _ in 0..reps {
            for p in 0..256u64 {
                hash_cast(&ctrl, p, 22, &mut cast);
            }
            for p in 0..256u64 {
                hash_cast(&ctrl, p, 14, &mut cast);
            }
            for p in 0..64u64 {
                hash_cast(&ctrl, p, 6, &mut cast);
            }
            std::hint::black_box(&cast);
        }
        let cast_secs = t.elapsed().as_secs_f64();

        // 2. Tree Z2 hashing: 254 + 254 + 62 slots (garbler; eval is ~same −k).
        let t = Instant::now();
        for _ in 0..reps {
            for j in 0..570u64 {
                std::hint::black_box(hash::hash_z2(&ctrl, j));
            }
        }
        let tree_secs = t.elapsed().as_secs_f64();

        // 3. Accumulator folds, PER-LEAF REFERENCE form: what the eval step
        //    would cost without the subtree/W-chain fold it actually uses —
        //    kept as the comparison point for that optimization.
        let mut acc_r = vec![[0u32; LAMBDA]; 8];
        let mut acc_t = vec![[0u32; LAMBDA]; 8];
        let t = Instant::now();
        for _ in 0..reps {
            for stage_k in [8u32, 8, 6] {
                for i in 0..stage_k {
                    let lz = i + 1;
                    for c in 0..(1u64 << (stage_k - lz)) {
                        let pp = c << lz;
                        for j in lz..=stage_k {
                            let ji = j as usize - 1;
                            wide_madd(&mut acc_r[ji], (pp & ((1u64 << j) - 1)) as u32, &cast);
                            wide_add(&mut acc_t[ji], &cast);
                        }
                    }
                }
            }
            std::hint::black_box(&acc_r);
        }
        let fold_secs = t.elapsed().as_secs_f64();

        // 4. Garbler residues_all on 256 casts (includes the to_vec copy).
        let casts = vec![[5u32; LAMBDA]; 256];
        let t = Instant::now();
        for _ in 0..reps {
            for _ in 0..3 {
                std::hint::black_box(residues_all(casts.clone()));
            }
        }
        let resid_secs = t.elapsed().as_secs_f64();

        let per = |s: f64| 1e3 * s / 30.0; // ms per bench_axb_stages rep (80 primes)
        eprintln!("per-rep (80 primes): cast hashing {:.2}ms | tree z2 {:.2}ms | eval folds {:.2}ms | garbler residues {:.2}ms",
            per(cast_secs), per(tree_secs), per(fold_secs), per(resid_secs));
    }
}
