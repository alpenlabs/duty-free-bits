//! Kernel-path chunk conversion and sub-chunk extraction: the straight-line
//! garble/eval of [`bin_to_word`](super::convert::bin_to_word) and
//! [`sub_chunk_extract`](super::convert::sub_chunk_extract), bypassing the
//! `System` (the [`super::fold`] pattern, extended to width-`l` wires).
//!
//! Both kernels are built from the same two moves:
//!
//! * **The one-hot doubling tree.** Level `m` holds the binary one-hot of the
//!   low `m` bits. The garbler hashes EVERY slot (its pass is x-blind); the
//!   evaluator hashes only the closed slots and solves the single open one per
//!   level through that level's 1-bit scale join:
//!   `y_hot = L_bit ⊕ diff ⊕ (⊕_{j≠hot} y_j)`.
//! * **Width-`l` casts + the pin join.** Every leaf is cast once to Z_{2^l}
//!   (`A_p = H(leaf_p)`, `l` CCRH blocks). The single width-`l` pin join
//!   `Σ_p A_p ⋈ 1` lets the evaluator solve the whole hot class by sums:
//!   `L_root = pin_diff` (the constant-1 wire's label is 0), so any linear
//!   functional `Σ c_p·A_p` with a constant coefficient `q` on the unsolved
//!   hot class evaluates as `Σ_{solved} c_p·A_p + q·(L_root − Σ_{solved} A_p)`.
//!
//! The extract kernel runs the fused `word_to_bin_up` schedule per sub-chunk:
//! residues `x mod 2^i` from per-level accumulators peel the bits LSB-first,
//! and the level-`k` functional `Σ p·A_p` is the peel upcast. Values are never
//! discovered — the evaluator knows `x`, so every hot index is index
//! arithmetic. No worklist, no `System`, no `Label` heap traffic in the inner
//! loops: Z₂ wires are `[u64; 2]` words and width-`l` wires are `[u32; λ]`
//! lane arrays.
//!
//! Nonce discipline (paper App. A, Def. 4): Z₂ tree hashes draw bulk-domain
//! ids and width-`l` cast hashes draw solo-domain ids, each from
//! caller-supplied fresh windows ([`chunk_kernel_nonces`],
//! [`extract_kernel_nonces`]); the garbler and evaluator share the window and
//! index it by slot position, so pads agree at every closed slot.

use crate::crypto::expand;
use crate::hash;
use crate::label::{CfLabel, LAMBDA, Label, delta_r};

/// Packed words of a CF Z₂ label (the fold-kernel working type).
pub(crate) type Z2 = [u64; 2];

/// One coordinate per u32 lane of a width-`l` CF label (2 ≤ l ≤ 32).
pub(crate) type Wide = [u32; LAMBDA];

// ---- bare-label helpers ----

#[inline]
fn z2_of(l: &Label) -> Z2 {
    let c = l.as_cf();
    debug_assert_eq!(c.modulus(), 2);
    let raw = c.raw_bits();
    [raw[0], raw[1]]
}

#[inline]
fn z2_label(w: Z2) -> Label {
    Label::Cf(CfLabel::from_raw_bits(vec![w[0], w[1]], 2))
}

#[inline]
fn wide_of(l: &Label) -> Wide {
    let c = l.as_cf();
    c.lanes().try_into().expect("CF lanes are LAMBDA wide")
}

#[inline]
fn wide_label(w: &Wide, modulus: u64) -> Label {
    let mask = (modulus - 1) as u32;
    Label::Cf(CfLabel::from_lanes(w.iter().map(|&l| l & mask).collect(), modulus))
}

/// `dst += src` lane-wise, mod 2^32.
///
/// The lane ops run UNMASKED: wrapping u32 arithmetic is a ring homomorphism
/// onto Z_{2^l} under truncation, so the `& (2^l − 1)` happens once at the
/// boundaries ([`wide_label`], diff emission, the peel shifts, [`peel_bit`])
/// instead of on every op.
#[inline]
fn wide_add(dst: &mut Wide, src: &Wide) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_add(*s);
    }
}

/// `dst −= src` lane-wise, mod 2^32 (see [`wide_add`] for masking).
#[inline]
fn wide_sub(dst: &mut Wide, src: &Wide) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_sub(*s);
    }
}

/// `dst += c·src` lane-wise, mod 2^32 (see [`wide_add`] for masking).
#[inline]
fn wide_madd(dst: &mut Wide, c: u32, src: &Wide) {
    if c == 0 {
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_add(c.wrapping_mul(*s));
    }
}

/// Pack bit `bit` of every lane into a Z₂ label's words.
#[inline]
fn pack_lane_bit(lanes: &Wide, bit: u32) -> Z2 {
    let mut out = [0u64; 2];
    for (w, chunk) in lanes.chunks_exact(64).enumerate() {
        let mut acc = 0u64;
        for (i, &l) in chunk.iter().enumerate() {
            acc |= (((l >> bit) & 1) as u64) << i;
        }
        out[w] = acc;
    }
    out
}

/// CCRH pad of one width-`l` cast switch, unpacked to lanes: the kernel twin
/// of [`hash::hash_solo`] for CF Z_{2^l} payloads on bare words. `nonce` is a
/// solo-domain id (bit 63 clear), globally fresh.
fn hash_cast(ctrl: &Z2, nonce: u64, l: u32, out: &mut Wide) {
    debug_assert!(nonce < (1u64 << 63), "solo nonce uses the bulk-domain bit");
    debug_assert!((2..=32).contains(&l));
    let mut seed = [0u8; 16];
    seed[0..8].copy_from_slice(&ctrl[0].to_le_bytes());
    seed[8..16].copy_from_slice(&ctrl[1].to_le_bytes());
    let words = (LAMBDA * l as usize).div_ceil(64);
    // 16 tail bytes keep the NEON gather's last 16-byte load in bounds.
    let mut buf = [0u8; 512 + 16];
    expand(seed, nonce, &mut buf[..words * 8]);
    #[cfg(target_arch = "aarch64")]
    if super::arena::neon_unpack_ok(l as usize) {
        super::arena::unpack_even_k_neon(&buf, l as usize, out);
        return;
    }
    super::arena::unpack_generic(&buf[..words * 8], l as usize, out);
}

/// `bit i` of `sub − res` as a Z₂ label: the peel step
/// `mod2k(div2k(sub − mod2k(res, k), i), 1)` on lanes. `sub` is width-k
/// lanes, `res` width-l lanes (l ≥ k).
#[inline]
fn peel_bit(sub: &Wide, res: &Wide, i: u32, k: u32) -> Z2 {
    let mask_k = ((1u64 << k) - 1) as u32;
    let mut out = [0u64; 2];
    for (w, (sc, rc)) in sub
        .chunks_exact(64)
        .zip(res.chunks_exact(64))
        .enumerate()
    {
        let mut acc = 0u64;
        for (idx, (&s, &r)) in sc.iter().zip(rc).enumerate() {
            let d = s.wrapping_sub(r) & mask_k;
            acc |= (((d >> i) & 1) as u64) << idx;
        }
        out[w] = acc;
    }
    out
}

// ---- cost / nonce accounting ----

/// Garbled-material + hash footprint of one kernel (the units `System::cost`
/// charges the equivalent circuit: a CF switch on Z_{2^w} costs `w` blocks, a
/// CF join `w` λ-bit units).
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelCost {
    /// Bits the garbler emits (join diffs).
    pub program_bits: usize,
    /// CF join width, in `lg|G|` units.
    pub join_complexity_cf: usize,
    /// CCRH blocks charged by the ledger (garbler-side: every switch).
    pub hash_count_cf: usize,
}

impl KernelCost {
    fn add(&mut self, o: KernelCost) {
        self.program_bits += o.program_bits;
        self.join_complexity_cf += o.join_complexity_cf;
        self.hash_count_cf += o.hash_count_cf;
    }
}

/// One tree-plus-casts stage: `n_bits` one-hot bits, casts at width `l`.
fn stage_cost(n_bits: u32, l: u32) -> KernelCost {
    let n = 1usize << n_bits;
    KernelCost {
        program_bits: (n_bits as usize - 1 + l as usize) * LAMBDA,
        join_complexity_cf: n_bits as usize - 1 + l as usize,
        hash_count_cf: (n - 2) + l as usize * n,
    }
}

/// Bulk-domain ids one tree consumes (the garbler hashes every slot of levels
/// 1..k−1; the evaluator indexes the same window by slot position).
#[inline]
fn tree_ids(k: u32) -> u64 {
    (1u64 << k) - 2
}

/// (bulk ids, solo ids) consumed by one chunk kernel over `n_bits` input bits.
pub fn chunk_kernel_nonces(n_bits: u32) -> (u64, u64) {
    (tree_ids(n_bits), 1u64 << n_bits)
}

/// (bulk ids, solo ids) consumed by one extract kernel.
pub fn extract_kernel_nonces(sub_widths: &[u32]) -> (u64, u64) {
    let mut bulk = 0;
    let mut solo = 0;
    for &w in sub_widths {
        bulk += tree_ids(w);
        solo += 1u64 << w;
    }
    (bulk, solo)
}

// ---- the shared tree/cast stage ----

/// Garbler side of one doubling tree: level-major, hashes EVERY slot.
/// Returns (leaf masks, per-level y-XOR sums `⊕_j y_m[j]` for levels 1..k−1).
///
/// `lvl1` is the level-1 pair `[X_{b0} ⊕ Δ₂, X_{b0}]` (the affine base
/// one-hot). Tree ids are `bulk_base + (2^m − 2) + j` for slot `j` of level
/// `m` — position-indexed so the evaluator lands on the same pads.
fn tree_garble(k: u32, lvl1: [Z2; 2], bulk_base: u64) -> (Vec<Z2>, Vec<Z2>) {
    let mut lvl: Vec<Z2> = lvl1.to_vec();
    let mut ysums: Vec<Z2> = Vec::with_capacity(k as usize - 1);
    for m in 1..k {
        let width = 1usize << m;
        let base = bulk_base + ((1u64 << m) - 2);
        let mut next = vec![[0u64; 2]; width * 2];
        let mut ysum = [0u64; 2];
        for j in 0..width {
            let y = hash::hash_z2(&lvl[j], base + j as u64);
            ysum[0] ^= y[0];
            ysum[1] ^= y[1];
            next[j] = [lvl[j][0] ^ y[0], lvl[j][1] ^ y[1]];
            next[j + width] = y;
        }
        ysums.push(ysum);
        lvl = next;
    }
    (lvl, ysums)
}

/// Width-`l` residue bank over a full cast array (garbler side, which holds
/// every cast): halving class sums with the incremental weighted chain
/// `res_{j+1} = res_j + 2^j·H_j`. Returns `(res_1..=res_k, root)`; `res_k` is
/// the upcast functional `Σ p·casts[p]`, `root = Σ_p casts[p]`. Consumes the
/// casts (halved in place).
fn residues_all(mut acc: Vec<Wide>) -> (Vec<Wide>, Wide) {
    let k = acc.len().trailing_zeros();
    let mut res_chain: Vec<Wide> = Vec::with_capacity(k as usize);
    // H_j for j = k−1 down to 1, gathered during halving.
    let mut highs: Vec<Wide> = Vec::with_capacity(k as usize);
    while acc.len() > 2 {
        let mid = acc.len() / 2;
        let mut high = [0u32; LAMBDA];
        for d in 0..mid {
            let (lo, hi) = acc.split_at_mut(mid);
            wide_add(&mut high, &hi[d]);
            wide_add(&mut lo[d], &hi[d]);
        }
        acc.truncate(mid);
        highs.push(high);
    }
    // Level 1: res_1 = S_1[1]; H_0-style entry not needed (bit 0 is native).
    let mut res = acc[1];
    res_chain.push(res);
    // res_{j+1} = res_j + 2^j·H_j, H_j in reverse gather order.
    for (j, high) in (1..k).zip(highs.iter().rev()) {
        let mut r = res;
        wide_madd(&mut r, 1u32 << j, high);
        res_chain.push(r);
        res = r;
    }
    let mut root = acc[0];
    wide_add(&mut root, &acc[1]);
    (res_chain, root)
}

// ---- extract kernel ----

/// Garbler-side output of one extract kernel.
#[derive(Debug)]
pub struct ExtractGarbleOutput {
    /// Masks of the first sub-chunk's binary one-hot (CF Z₂, 2^{w₀} entries).
    pub first_bin_hot_masks: Vec<Label>,
    /// Masks of the fold bits (sub-chunks 1.., LSB-first within each).
    pub fold_bit_masks: Vec<Label>,
    /// Per sub-chunk: the k−1 scale-join diffs and the width-`l` pin diff.
    pub diffs: Vec<SubChunkDiffs>,
    /// Ledger footprint.
    pub cost: KernelCost,
}

/// The communicated material of one sub-chunk stage.
#[derive(Debug)]
pub struct SubChunkDiffs {
    /// Scale-join diff per tree level (k−1 of them, Z₂).
    pub scale: Vec<Z2>,
    /// Pin-join diff (width `l`).
    pub pin: Wide,
    /// Cast width `l` of this stage.
    pub l: u32,
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

/// Garbler kernel: `r = Σ_c coeff_c·w_c`, then the fused `word_to_bin_up`
/// per sub-chunk. X-blind: hashes every tree slot and every cast, including
/// the (unknown) hot ones.
///
/// `chunk_word_masks` are CF Z_{2^ell} masks of the chunk words; `nonce_bulk`
/// / `nonce_solo` are fresh windows sized by [`extract_kernel_nonces`].
pub fn extract_batch_garble(
    chunk_word_masks: &[Label],
    coeffs: &[u64],
    sub_widths: &[u32],
    delta: u128,
    nonce_bulk: u64,
    nonce_solo: u64,
) -> ExtractGarbleOutput {
    let ell = chunk_word_masks[0].as_cf().k();
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
    let mut cost = KernelCost::default();
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
        wide_add(&mut pin, &wide_of(&Label::Cf(dl)));
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

        diffs.push(SubChunkDiffs { scale, pin, l });
        cost.add(stage_cost(k, l));
    }

    ExtractGarbleOutput {
        first_bin_hot_masks,
        fold_bit_masks,
        diffs,
        cost,
    }
}

/// Evaluator kernel: the label-side mirror of [`extract_batch_garble`],
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
    let ell = chunk_word_labels[0].as_cf().k();
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

// ---- chunk kernel (bin_to_word) ----

/// Garbler-side output of one chunk-conversion kernel.
#[derive(Debug)]
pub struct ChunkGarbleOutput {
    /// Mask of the chunk word (CF Z_{2^ell}).
    pub word_mask: Label,
    /// The n−1 scale-join diffs and the width-ell pin diff.
    pub scale: Vec<Z2>,
    /// Pin-join diff (width ell).
    pub pin: Wide,
    /// Ledger footprint.
    pub cost: KernelCost,
}

/// Garbler kernel: pack `bits` (CF Z₂ masks, LSB-first) into a Z_{2^ell}
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
    for (pp, cast) in casts.iter_mut().enumerate() {
        hash_cast(&leaves[pp], nonce_solo + pp as u64, ell, cast);
    }
    let (res_chain, root) = residues_all(casts);
    let word_mask = res_chain[nb as usize - 1]; // Σ p·A_p

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
    wide_add(&mut pin, &wide_of(&Label::Cf(dl)));
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

/// Evaluator kernel: the label-side mirror of [`chunk_batch_garble`].
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
    for (pp, leaf) in lvl.iter().enumerate() {
        if pp == hot {
            continue;
        }
        hash_cast(leaf, nonce_solo + pp as u64, ell, &mut cast);
        wide_madd(&mut word, pp as u32, &cast);
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
        Label::Cf(CfLabel::from_coords(&coords, 2))
    }

    fn rand_wide(rng: &mut impl Rng, modulus: u64) -> Label {
        let coords: Vec<u64> = (0..LAMBDA)
            .map(|_| rng.random_range(0..modulus))
            .collect();
        Label::Cf(CfLabel::from_coords(&coords, modulus))
    }

    /// Chunk kernel: word label = word mask + v·Δ for every value.
    #[test]
    fn test_chunk_kernel_label_mask_invariant() {
        let mut rng = rand::rng();
        let delta: u128 = rng.random::<u128>() | 1;
        let ell = 12u32;
        let nb = 4u32;
        let d2 = Label::Cf(delta_r(delta, 2));
        let dl = Label::Cf(delta_r(delta, 1u64 << ell));

        let bit_masks: Vec<Label> = (0..nb).map(|_| rand_z2(&mut rng)).collect();
        let g = chunk_batch_garble(&bit_masks, ell, delta, 5_000, 9_000);
        assert_eq!(g.cost.join_complexity_cf as u32, nb - 1 + ell);
        assert_eq!(
            g.cost.hash_count_cf as u64,
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
            assert_eq!(w, expect, "chunk word label ≠ mask + v·Δ for v={v}");
        }
    }

    /// Extract kernel: OHE labels + fold-bit labels satisfy the carry
    /// invariant for every r, at the production shape and a small one.
    #[test]
    fn test_extract_kernel_label_mask_invariant() {
        let mut rng = rand::rng();
        let delta: u128 = rng.random::<u128>() | 1;
        let d2 = Label::Cf(delta_r(delta, 2));

        for (ell, sub_widths, coeffs) in [
            (8u32, vec![4u32, 4], vec![1u64, 16]),
            (22, vec![8, 8, 6], vec![1, 37, 4123]),
        ] {
            let m_ell = 1u64 << ell;
            let word_masks: Vec<Label> =
                (0..coeffs.len()).map(|_| rand_wide(&mut rng, m_ell)).collect();
            let (bulk_n, solo_n) = extract_kernel_nonces(&sub_widths);

            let g = extract_batch_garble(&word_masks, &coeffs, &sub_widths, delta, 100, 200);
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
                let dl_ell = Label::Cf(delta_r(delta, m_ell));
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

    /// Attribute extract-kernel time to its primitives at production shape.
    /// Run: cargo test --release --lib bench_extract_micro -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_extract_micro() {
        let reps = 80 * 30; // 80 primes × 30 reps, i.e. "per bench_kernel_loop run"
        let ctrl: Z2 = [0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321];
        let mut cast = [7u32; LAMBDA];
        let mask22 = (1u32 << 22) - 1;

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

        // 3. Accumulator folds, PER-LEAF REFERENCE form: what the eval kernel
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

        let per = |s: f64| 1e3 * s / 30.0; // ms per bench_kernel_loop rep (80 primes)
        eprintln!("per-rep (80 primes): cast hashing {:.2}ms | tree z2 {:.2}ms | eval folds {:.2}ms | garbler residues {:.2}ms",
            per(cast_secs), per(tree_secs), per(fold_secs), per(resid_secs));
    }
}
