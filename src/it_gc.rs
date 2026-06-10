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

    // Σ pad_i and Σ g_i·pad_i per member, delayed reduction (see accumulate_pads).
    let mut pad_sum_raw = vec![0u64; b]; // Σ_i pad_i (unreduced)
    let mut readout_raw = vec![0u64; b]; // Σ_i g_i·pad_i (unreduced)
    accumulate_pads(
        &slot_hash,
        truth_table,
        None,
        p_i,
        &mut pad_sum_raw,
        &mut readout_raw,
    );

    let (join_diffs, result_masks) =
        garble_outputs_from_sums(p_i, a_batch, b_batch, &pad_sum_raw, &readout_raw);

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

    // Σ pads over every slot but `hot`, delayed reduction (see accumulate_pads).
    let mut pad_sum_raw = vec![0u64; b]; // Σ_{i≠hot} pad_i  (every pad we can recompute)
    let mut readout_raw = vec![0u64; b]; // Σ_{i≠hot} g(i)·pad_i
    accumulate_pads(
        &slot_hash,
        truth_table,
        Some(hot),
        p_i,
        &mut pad_sum_raw,
        &mut readout_raw,
    );

    eval_outputs_from_sums(p_i, g_hot, join_diffs, &pad_sum_raw, &readout_raw)
}

/// Fold the garbler's unreduced pad sums into the per-member outputs.
fn garble_outputs_from_sums(
    p_i: u64,
    a_batch: &[u64],
    b_batch: &[u64],
    pad_sum_raw: &[u64],
    readout_raw: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let b = a_batch.len();
    let mut join_diffs = Vec::with_capacity(b);
    let mut result_masks = Vec::with_capacity(b);
    for j in 0..b {
        let pad_sum = pad_sum_raw[j] % p_i;
        let readout = mod_add(readout_raw[j] % p_i, neg_mod(b_batch[j], p_i), p_i);
        // The one residue sent per map: diff = Σ pad_i + a (mod_sub by −a adds a).
        join_diffs.push(mod_sub(pad_sum, neg_mod(a_batch[j], p_i), p_i));
        result_masks.push(readout);
    }
    (join_diffs, result_masks)
}

/// Fold the evaluator's unreduced pad sums into the delivered residues.
fn eval_outputs_from_sums(
    p_i: u64,
    g_hot: u64,
    join_diffs: &[u64],
    pad_sum_raw: &[u64],
    readout_raw: &[u64],
) -> Vec<u64> {
    join_diffs
        .iter()
        .enumerate()
        .map(|(j, &diff)| {
            // Open the one hidden pad; the diff leaves `a` sitting on it:
            //   pad_hot + a = diff − Σ_{i≠hot} pad_i.
            let pad_hot = mod_sub(diff, pad_sum_raw[j] % p_i, p_i);
            mod_add(readout_raw[j] % p_i, mod_mul(g_hot, pad_hot, p_i), p_i)
        })
        .collect()
}

/// Slot-major slab of the per-slot bulk hashes: slot `i`'s `b·lg|p_i|` pad
/// bits occupy `bytes[i·stride .. i·stride + exact_len]`, where `stride` is
/// `exact_len` rounded up to a u64 boundary. 8 trailing zero bytes guarantee
/// that any u64 load starting inside a slot's exact region stays in bounds.
struct HashSlab {
    bytes: Vec<u8>,
    stride: usize,
}

/// The pad material `H(h_p[i])` for each one-hot slot, bulk-packed: each slot
/// holds `b·lg|p_i|` pseudorandom bits, one `lg|p_i|`-bit pad per batch member.
/// Garbler and evaluator call this identically (on masks / labels); at every
/// non-hot slot the two agree, which is what lets `pad_i` line up.
fn switch_hashes(ohe: &[Label], group_id_base: usize, b: usize, p_i: u64) -> HashSlab {
    let lg_p = hash::lg_modulus(p_i);
    let total_bits = b * lg_p;
    let exact_len = total_bits.div_ceil(8);
    let stride = exact_len.div_ceil(8) * 8;
    // One allocation for the whole batch; intra-slot padding and the 8-byte
    // tail stay zero (and are masked off in `extract_pad` regardless).
    let mut bytes = vec![0u8; ohe.len() * stride + 8];
    for (i, l) in ohe.iter().enumerate() {
        let base = i * stride;
        hash::hash_bulk_into(
            l,
            group_id_base + i,
            total_bits,
            &mut bytes[base..base + exact_len],
        );
    }
    HashSlab { bytes, stride }
}

/// Accumulate `Σ_i pad_i` and `Σ_i g_i·pad_i` per batch member into
/// `pad_sum_raw` / `readout_raw`, skipping slot `skip` entirely when given
/// (eval's hot slot — always a whole-slot skip, never a per-member one).
///
/// Delayed reduction: pads and weights are both < p, so with p³ < 2⁶³ the u64
/// sums `Σ pad_i` (< p²) and `Σ g_i·pad_i` (< p³, at most p terms) cannot
/// overflow — one `% p` per member at the fold replaces one per slot.
///
/// Dispatches to the NEON kernel on aarch64 when the modulus fits its u32
/// accumulators (see the gate below); the scalar kernel is both the portable
/// path and the fallback. The kernels produce bit-identical sums: each pad is
/// the same exact integer either way and no addition can overflow, so the
/// summation order is immaterial.
fn accumulate_pads(
    slab: &HashSlab,
    truth_table: &[u64],
    skip: Option<usize>,
    p_i: u64,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    debug_assert!(
        (p_i as u128) * (p_i as u128) * (p_i as u128) < (1u128 << 63),
        "delayed-reduction accumulators would overflow for p = {p_i}"
    );
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // u32-accumulator bound, b-independent: readout_raw[j] sums at most
        // p terms (one per slot), each g·pad < p², so the total is < p³; u32
        // is exact iff p³ < 2³², i.e. p ≤ 1625. That bound also caps
        // lg_p ≤ 11, which the NEON kernel's single-u64 group load needs.
        // p ≥ 2 keeps the slab's exact region non-empty (lg_p ≥ 1).
        let p128 = p_i as u128;
        if p_i >= 2 && p128 * p128 * p128 < (1u128 << 32) {
            accumulate_pads_neon(slab, truth_table, skip, p_i, pad_sum_raw, readout_raw);
            return;
        }
    }
    accumulate_pads_scalar(slab, truth_table, skip, p_i, 0, pad_sum_raw, readout_raw);
}

/// Portable pad accumulation: slot-major walk, one [`extract_pad`] per
/// (slot, member). Covers members `j_lo..` only — the NEON kernel reuses it
/// for the ragged tail after its 4-member groups.
fn accumulate_pads_scalar(
    slab: &HashSlab,
    truth_table: &[u64],
    skip: Option<usize>,
    p_i: u64,
    j_lo: usize,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    let lg_p = hash::lg_modulus(p_i);
    for (i, &t) in truth_table.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let g = t % p_i;
        let base = i * slab.stride;
        for (j, (ps, ro)) in pad_sum_raw
            .iter_mut()
            .zip(readout_raw.iter_mut())
            .enumerate()
            .skip(j_lo)
        {
            let pad = extract_pad(&slab.bytes, base, j, lg_p, p_i);
            *ps += pad;
            *ro += g * pad;
        }
    }
}

/// NEON pad accumulation: 4 members per u32 lane group, member-major so each
/// group's accumulators stay in registers across the whole slot walk.
///
/// Caller-guaranteed bounds (the dispatch gate): `2 ≤ p` and `p³ < 2³²`. Then
/// * the u32 lanes are exact — pad-sum lanes stay < p² and readout lanes
///   < p³ < 2³² across all ≤ p slots, so no lane ever wraps and one widening
///   into the u64 sums per group reproduces the scalar totals bit for bit;
/// * lg_p ≤ 11, so a group's four pads span ≤ 7 + 4·11 = 51 bits of one
///   unaligned u64 load (≤ 7 is the load's bit phase, `shift0`).
///
/// Per lane this is the vector transcription of [`extract_pad`]: right-shift
/// the group word so the lane's pad starts at bit 0 (vshlq by a negative
/// count), mask to lg_p bits, one compare-subtract — identical pad values.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn accumulate_pads_neon(
    slab: &HashSlab,
    truth_table: &[u64],
    skip: Option<usize>,
    p_i: u64,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    use core::arch::aarch64::*;

    let b = pad_sum_raw.len();
    let lg_p = hash::lg_modulus(p_i);
    debug_assert!((1..=11).contains(&lg_p), "NEON gate admits 2 ≤ p ≤ 1625");
    // Per-slot weights g = truth_table[i] mod p, hoisted out of the
    // member-major walk (g < p ≤ 1625 fits u32 by the gate).
    let gs: Vec<u32> = truth_table.iter().map(|&t| (t % p_i) as u32).collect();
    let full = b & !3; // members [0, full) in groups of 4; tail goes scalar.

    // SAFETY: NEON intrinsics — the cfg above pins target_arch + feature. The
    // u64 group loads start at `i·stride + (j0·lg_p >> 3)` with `j0 + 4 ≤ b`,
    // i.e. inside slot i's exact region (j0·lg_p < total_bits), which the
    // slab's 8 trailing bytes keep in bounds for an 8-byte read (see HashSlab).
    unsafe {
        let p_v = vdupq_n_u32(p_i as u32);
        let mask_v = vdupq_n_u32((1u32 << lg_p) - 1);
        for j0 in (0..full).step_by(4) {
            let bit_off = j0 * lg_p;
            let byte_off = bit_off >> 3;
            let shift0 = (bit_off & 7) as i64;
            // vshlq with a negative count right-shifts: lane k's count drops
            // the bits below its pad, [shift0 + k·lg_p ..][..lg_p].
            let shifts = [0i64, 1, 2, 3].map(|k| -(shift0 + k * lg_p as i64));
            let sh01 = vld1q_s64(shifts.as_ptr());
            let sh23 = vld1q_s64(shifts.as_ptr().add(2));

            let mut ps = vdupq_n_u32(0);
            let mut ro = vdupq_n_u32(0);
            for (i, &g) in gs.iter().enumerate() {
                if Some(i) == skip {
                    continue;
                }
                let ptr = slab.bytes.as_ptr().add(i * slab.stride + byte_off);
                let raw = u64::from_le_bytes(core::ptr::read_unaligned(ptr.cast::<[u8; 8]>()));
                let v = vdupq_n_u64(raw);
                let acc = vandq_u32(
                    vcombine_u32(vmovn_u64(vshlq_u64(v, sh01)), vmovn_u64(vshlq_u64(v, sh23))),
                    mask_v,
                );
                // acc < 2^⌈log₂ p⌉ < 2p, so subtracting p exactly where
                // acc ≥ p matches extract_pad's compare-subtract.
                let pad = vsubq_u32(acc, vandq_u32(vcgeq_u32(acc, p_v), p_v));
                ps = vaddq_u32(ps, pad);
                ro = vmlaq_n_u32(ro, pad, g);
            }
            let mut ps_arr = [0u32; 4];
            let mut ro_arr = [0u32; 4];
            vst1q_u32(ps_arr.as_mut_ptr(), ps);
            vst1q_u32(ro_arr.as_mut_ptr(), ro);
            for k in 0..4 {
                pad_sum_raw[j0 + k] += ps_arr[k] as u64;
                readout_raw[j0 + k] += ro_arr[k] as u64;
            }
        }
    }
    // Ragged tail (b mod 4 members): scalar kernel, identical pads.
    if full < b {
        accumulate_pads_scalar(slab, truth_table, skip, p_i, full, pad_sum_raw, readout_raw);
    }
}

/// Member `j`'s pad `pad_i` (an NCF Z_p residue) sliced from slot `i`'s bulk
/// hash: bits `[j·lg_p .. (j+1)·lg_p)` past `base = i·stride`, reduced into Z_p.
///
/// Computes the same value as [`hash::extract_ncf`] (LSB-first bit slice, then
/// `% p`), but via a single unconditional u64 load instead of a per-bit loop —
/// the slab's 8-byte tail keeps the load in bounds, and the `lg_p`-bit mask
/// keeps padding bytes out of the value — and a compare-subtract instead of a
/// division (`acc < 2^⌈lg p⌉ < 2p`).
#[inline(always)]
fn extract_pad(slab: &[u8], base: usize, member_idx: usize, lg_p: usize, p: u64) -> u64 {
    debug_assert!(lg_p <= 57, "extract_pad u64 load covers ≤ 57-bit slices");
    let bit_off = member_idx * lg_p;
    let byte = base + (bit_off >> 3);
    let raw = u64::from_le_bytes(slab[byte..byte + 8].try_into().unwrap());
    let acc = (raw >> (bit_off & 7)) & ((1u64 << lg_p) - 1);
    // acc < 2^⌈log₂ p⌉ < 2p, so one conditional subtract equals `acc % p`.
    if acc >= p { acc - p } else { acc }
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
    use rand::Rng;

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

    /// Differential test for the SIMD dispatch: the auto kernel (NEON on
    /// aarch64 wherever the p³ < 2³² gate admits it, scalar everywhere else —
    /// every p below passes the gate) against the always-scalar kernel, end
    /// to end. Raw accumulator sums, garbler outputs (join diffs + result
    /// masks) and eval outputs must be bit-identical, across power-of-two and
    /// odd moduli, ragged batch sizes, and a sweep of hot positions.
    #[test]
    fn test_body_batch_simd_matches_scalar_differential() {
        let mut rng = rand::rng();
        let moduli: [u64; 14] = [2, 3, 5, 7, 17, 31, 127, 128, 129, 251, 257, 409, 1009, 1621];
        let batch_sizes: [usize; 12] = [1, 2, 3, 7, 8, 15, 16, 31, 64, 127, 128, 129];
        let gid = 17usize;
        for &p_i in &moduli {
            let p = p_i as usize;
            let truth: Vec<u64> = (0..p_i).collect();
            let mut hots = vec![0usize, 1 % p, p / 2, p - 1];
            hots.sort_unstable();
            hots.dedup();
            for &b in &batch_sizes {
                let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
                let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
                let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();

                // Garbler: full kernel (auto dispatch) vs the forced-scalar
                // reference rebuilt from the same deterministic slab.
                let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &truth, gid);
                let slab = switch_hashes(&h_p_masks, gid, b, p_i);
                let (mut ps_auto, mut ro_auto) = (vec![0u64; b], vec![0u64; b]);
                accumulate_pads(&slab, &truth, None, p_i, &mut ps_auto, &mut ro_auto);
                let (mut ps_ref, mut ro_ref) = (vec![0u64; b], vec![0u64; b]);
                accumulate_pads_scalar(&slab, &truth, None, p_i, 0, &mut ps_ref, &mut ro_ref);
                assert_eq!(ps_auto, ps_ref, "pad sums diverge: p={p_i} b={b}");
                assert_eq!(ro_auto, ro_ref, "readouts diverge: p={p_i} b={b}");
                let (jd_ref, rm_ref) =
                    garble_outputs_from_sums(p_i, &a_batch, &b_batch, &ps_ref, &ro_ref);
                assert_eq!(
                    g_out.join_diffs, jd_ref,
                    "join diffs diverge: p={p_i} b={b}"
                );
                assert_eq!(
                    g_out.result_masks, rm_ref,
                    "result masks diverge: p={p_i} b={b}"
                );

                for &hot in &hots {
                    // Labels equal the masks except the LSB flip at the hot slot.
                    let h_p_labels: Vec<Label> = h_p_masks
                        .iter()
                        .enumerate()
                        .map(|(i, m)| {
                            if i == hot {
                                let mut coords = m.as_cf().to_coords();
                                coords[0] ^= 1;
                                Label::Cf(CfLabel::from_coords(&coords, 2))
                            } else {
                                m.clone()
                            }
                        })
                        .collect();
                    let ev = body_batch_eval(
                        p_i,
                        hot,
                        &h_p_labels,
                        &g_out.join_diffs,
                        &b_batch,
                        &truth,
                        gid,
                    );
                    let slab_l = switch_hashes(&h_p_labels, gid, b, p_i);
                    let (mut ps_e, mut ro_e) = (vec![0u64; b], vec![0u64; b]);
                    accumulate_pads_scalar(
                        &slab_l,
                        &truth,
                        Some(hot),
                        p_i,
                        0,
                        &mut ps_e,
                        &mut ro_e,
                    );
                    let ev_ref = eval_outputs_from_sums(
                        p_i,
                        truth[hot] % p_i,
                        &g_out.join_diffs,
                        &ps_e,
                        &ro_e,
                    );
                    assert_eq!(ev, ev_ref, "eval outputs diverge: p={p_i} b={b} hot={hot}");
                    // And the protocol still decodes b + hot·a per member.
                    for j in 0..b {
                        let value = mod_sub(ev[j], g_out.result_masks[j], p_i);
                        let expected = (b_batch[j] + (hot as u64) * a_batch[j]) % p_i;
                        assert_eq!(value, expected, "round trip: p={p_i} b={b} hot={hot} j={j}");
                    }
                }
            }
        }
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
