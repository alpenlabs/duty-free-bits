//! Step 4 / 4 (information-theoretic GC): the canonical **one-hot scaling** —
//! deliver `a·(x mod p_i) + b` per prime.
//!
//! Step 3 ([`super::fold`]) handed us a length-`p_i` one-hot `h_p` of
//! `hot = x mod p_i`: `p_i` boolean labels, exactly the one at the active slot
//! `hot` sharing a 1. Hashing slot `i`'s label gives a fresh one-time pad
//! `pad_i = H(h_p[i]) ∈ Z_{p_i}` — one pad per slot, rejection-sampled from
//! the slot's hash stream so it is exactly uniform (see [`slot_pads`]).
//!
//! **One pad is hidden from the evaluator.** The evaluator knows `x`, hence the
//! active slot `hot`. At every *off* slot the one-hot bit is 0, so its label
//! equals the garbler's mask and the evaluator recomputes `pad_i` itself. At the
//! *active* slot the label differs, so `pad_hot` is the single pad it cannot
//! derive. The `p_i` pads are therefore an additive sharing the evaluator can
//! open everywhere but that one slot.
//!
//! **Delivering `a` — the scaling ciphertext.** The garbler sends one residue
//! per affine map
//!
//! ```text
//!     diff = (Σ_i pad_i) + a            (mod p_i)
//! ```
//!
//! The evaluator subtracts every pad it knows: `pad_hot + a = diff − Σ_{i≠hot} pad_i`.
//! Now the active slot carries `pad_hot + a` and every off slot carries plain
//! `pad_i` — i.e. the scalar `a` sits at the active slot, and every off slot shares 0.
//!
//! A **free** recombination then weights slot `i` by `i` (and folds in `b`),
//! turning the `a` at the active slot into `a·(x mod p_i) + b`.
//! `(a, b)` stay hidden information-theoretically: single `Z_p` residues.

use crate::gc::Cost;
use crate::hash;
use crate::label::{LAMBDA, Label};

/// Garbler-side output of one body batch, length `B` (one entry per member).
///
/// The evaluator holds `x` in the clear, so it knows every active position
/// itself; the garbler emits only the scaling ciphertexts (`join_diffs`).
#[derive(Debug)]
pub struct BodyBatchGarbleOutput {
    /// `join_diffs` — the one residue each scaling communicates per batch member:
    /// `Σ pad_i + a` as a `Z_p` share.
    pub join_diffs: Vec<u64>,
    /// Garbler's output mask per batch member (`Z_p` share).
    pub result_masks: Vec<u64>,
    /// Garbled-material footprint of this batch, for telemetry.
    pub cost: Cost,
}

/// Garbler side of one per-prime body batch (see the module docs): forms the
/// per-slot pads, sends `diff = Σ pad_i + a` per member.
///
/// `h_p_masks` are the carry-in boolean-label masks for the length-`p_i`
/// one-hot; `a_batch` / `b_batch` are the affine coefficients reduced mod `p_i`;
/// `weights[i] = i mod p_i` is the recombination weight `g`. `group_id_base`
/// offsets the per-slot CCRH tweak.
pub fn body_batch_garble(
    p_i: u64,
    h_p_masks: &[Label],
    a_batch: &[u64],
    b_batch: &[u64],
    weights: &[u64],
    group_id_base: usize,
) -> BodyBatchGarbleOutput {
    let p = p_i as usize;
    let b = a_batch.len();
    assert_eq!(h_p_masks.len(), p, "h_p_masks length mismatch");
    assert_eq!(b_batch.len(), b, "a_batch and b_batch length mismatch");
    assert_eq!(weights.len(), p, "weights length mismatch");

    // pad_i = H(h_p[i]) for each one-hot slot, bulk-packed across the b members.
    // The garbler is x-blind, so it samples every slot's stream.
    let slot_hash = slot_pads(h_p_masks, group_id_base, b, p_i, None);

    // Delayed reduction — see accumulate_pads.
    let mut pad_sum_raw = vec![0u64; b]; // Σ_i pad_i (unreduced)
    let mut readout_raw = vec![0u64; b]; // Σ_i g_i·pad_i (unreduced)
    accumulate_pads(
        &slot_hash,
        weights,
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

/// Garbled-material footprint of a `b`-member body batch mod `p_i`: one scaling
/// residue of `lg|p_i|` bits per member (communication is unchanged by the pad
/// sampling), and the EXPECTED CCRH block count of the `p_i` slot streams.
///
/// With rejection sampling ([`slot_pads`]) the realized block count is random
/// — each slot scans a negative-binomially distributed number of `w`-bit
/// windows to collect its `b` accepts — so the ledger reports a deterministic
/// estimate: `b·w / (1 − r) = b·w·2^w / acc` expected bits per slot
/// (`acc` = [`accept_bound`], `r` the rejection rate of [`pad_width`]'s rule),
/// plus λ/2 for drawing the stream in whole blocks, all over λ. When `r = 0`
/// (p = 2 in the concrete prime set: `acc = 2^w`) no window can reject and the
/// exact deterministic count is reported instead. `bench_axb_hashcounts`
/// cross-checks the measured garbler count against this to ±3%; on the
/// 80-prime set the estimate sits within ~0.1% of the true expectation.
fn batch_cost(p_i: u64, b: usize) -> Cost {
    let join_bits = b * hash::lg_modulus(p_i);
    let w = pad_width(p_i);
    let acc = accept_bound(p_i, w);
    let (bw, lambda) = ((b * w) as u64, LAMBDA as u64);
    let hash_count = if acc == 1u64 << w {
        p_i * bw.div_ceil(lambda)
    } else {
        // p·(b·w·2^w/(acc·λ) + 1/2), rounded up in exact integer arithmetic
        // (< 2^36 at the largest test modulus — no overflow near u64).
        (p_i * (2 * bw * (1u64 << w) + acc * lambda)).div_ceil(2 * acc * lambda)
    };
    Cost {
        program_bits: join_bits,
        join_complexity: join_bits,
        hash_count: hash_count as usize,
    }
}

/// Evaluator side of one per-prime body batch: the label-side mirror of
/// [`body_batch_garble`] (see the module docs).
///
/// `hot = x mod p_i` is supplied in the clear — the evaluator holds `x`, so it
/// knows the active position without it ever being revealed. It recomputes every
/// pad but `pad_hot`, opens the hidden one from `diff`, and reads out the result.
/// (`b_batch` is unused beyond its length — `b`'s label is 0.) Returns the
/// delivered residue per member; the caller decodes `value = label − mask mod p_i`.
pub fn body_batch_eval(
    p_i: u64,
    hot: usize,
    h_p_labels: &[Label],
    join_diffs: &[u64],
    b_batch: &[u64],
    weights: &[u64],
    group_id_base: usize,
) -> Vec<u64> {
    let p = p_i as usize;
    let b = join_diffs.len();
    assert_eq!(h_p_labels.len(), p, "h_p_labels length mismatch");
    assert_eq!(b_batch.len(), b, "b_batch length mismatch");
    assert_eq!(weights.len(), p, "weights length mismatch");
    assert!(hot < p, "hot index out of range");

    // Same pads the garbler formed. At an off slot the one-hot bit is 0, so
    // label == mask and the evaluator recomputes pad_i exactly. At the active
    // slot its label diverges — the stream (and its accept/reject pattern)
    // would be garbage — so it skips that slot outright and opens pad_hot
    // from `diff` below.
    let slot_hash = slot_pads(h_p_labels, group_id_base, b, p_i, Some(hot));
    let g_hot = weights[hot] % p_i;

    // Delayed reduction — see accumulate_pads.
    let mut pad_sum_raw = vec![0u64; b]; // Σ_{i≠hot} pad_i  (every pad we can recompute)
    let mut readout_raw = vec![0u64; b]; // Σ_{i≠hot} g(i)·pad_i
    accumulate_pads(
        &slot_hash,
        weights,
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
        // The scaling ciphertext, one residue per map: diff = Σ pad_i + a (mod_sub by −a adds a).
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

/// Bits of hash output sliced per sampling attempt: the window width
/// `w ∈ {4, 8, 12, 16}` minimizing the expected hash bits per accepted pad,
/// `w / (1 − r(w))` with rejection rate `r(w) = (2^w mod p) / 2^w` (ties break
/// to the smaller width; widths with `2^w < p`, where every window rejects,
/// never win).
///
/// Rejection against [`accept_bound`] makes the pads EXACTLY uniform over
/// `Z_p` — see [`slot_pads`] — where reducing a bare `w`-bit slice mod `p` is
/// biased by `r(w)` (up to ~49% at p = 131 for the former nibble-aligned
/// `w = 8`). The rule prices that exactness per prime: a wider window costs
/// more bits per attempt but rejects less often. Concretely, on the 80-prime
/// set: `w = 4` for p ≤ 13, `w = 8` for 17 ≤ p ≤ 128 and 167 ≤ p ≤ 251,
/// `w = 12` for 131 ≤ p ≤ 163 and 257 ≤ p ≤ 409 (`w = 16` never wins below
/// `p = 2^12`); `test_pad_width_rule` pins the rule and these bands.
///
/// Every candidate width is a nibble multiple, so window extraction and slab
/// packing stay load + shift + mask on bit phases {0, 4} with no per-pad
/// reduction ([`extract_pad`] / [`pack_pad`]). The scaling residues — the
/// actual communication — remain `lg|p_i|`-bit.
fn pad_width(p_i: u64) -> usize {
    assert!(
        (2..=1 << 16).contains(&p_i),
        "pad_width covers 2 ≤ p ≤ 2^16 (the concrete primes are ≤ 409)"
    );
    let mut best: Option<(usize, u64, u64)> = None; // (w, w·2^w, accept_bound)
    for w in [4usize, 8, 12, 16] {
        let acc = accept_bound(p_i, w);
        if acc == 0 {
            continue;
        }
        // Minimize w·2^w / acc, compared as cross products (≤ 16·2^32 ≪ u64).
        let num = (w as u64) << w;
        match best {
            Some((_, bn, bd)) if num * bd >= bn * acc => {}
            _ => best = Some((w, num, acc)),
        }
    }
    best.expect("some window width admits acceptance").0
}

/// Acceptance bound of the `w`-bit rejection sampler for `Z_p`: windows
/// `v < p·⌊2^w/p⌋` are accepted — an exact multiple of `p` equiprobable
/// values, so `v mod p` is exactly uniform — and the `2^w mod p` values above
/// are rejected. Zero (nothing accepted) iff `2^w < p`.
fn accept_bound(p_i: u64, w: usize) -> u64 {
    ((1u64 << w) / p_i) * p_i
}

/// Slot-major slab of the per-slot accepted pads: slot `i`'s `b·pad_width`
/// pad bits occupy `bytes[i·stride .. i·stride + exact_len]`, where `stride`
/// is `exact_len` rounded up to a u64 boundary. 8 trailing zero bytes
/// guarantee that any u64 load starting inside a slot's exact region stays in
/// bounds.
struct HashSlab {
    bytes: Vec<u8>,
    stride: usize,
}

/// The pad material for each one-hot slot, bulk-packed: each slot holds `b`
/// accepted `w`-bit windows of its CCRH stream `H(h_p[i])`, one pad per batch
/// member (`w` = [`pad_width`]).
///
/// Per slot, successive `w`-bit windows are read off the stream; the first
/// window below [`accept_bound`] becomes the next member's pad (its residue
/// mod `p` is exactly uniform), rejected windows are skipped, and retries are
/// unbounded — a geometric tail, no fallback. The stream starts at the
/// reject-free footprint (`⌈b·w/λ⌉` blocks, one bulk CCRH call) and extends
/// one block at a time via [`hash::hash_bulk_more`] under the slot's one
/// nonce: the CTR block counter, not the nonce, sequences a stream, so the
/// [`crate::affine`] nonce layout is untouched by rejections. Accepted
/// windows are packed RAW (not reduced): the fold's single `% p` per member
/// absorbs the reduction, exactly as before (`Σ vᵢ ≡ Σ (vᵢ mod p)`).
///
/// Garbler and evaluator call this identically (on masks / labels): at every
/// off slot the two streams agree bit for bit, so the accept/reject pattern —
/// and hence every pad — lines up. At the active slot the evaluator's stream
/// diverges and its pads would be garbage; it passes `skip` to neither hash
/// nor scan that slot (its slab region stays zero, and [`accumulate_pads`]
/// skips it regardless), which is why the evaluator's measured block count is
/// the garbler's minus the active slots'. The garbler passes `None`.
fn slot_pads(
    ohe: &[Label],
    group_id_base: usize,
    b: usize,
    p_i: u64,
    skip: Option<usize>,
) -> HashSlab {
    let w = pad_width(p_i);
    let bound = accept_bound(p_i, w);
    let total_bits = b * w;
    let exact_len = total_bits.div_ceil(8);
    let stride = exact_len.div_ceil(8) * 8;
    // One allocation for the whole batch; intra-slot padding, a skipped slot
    // and the 8-byte tail stay zero (and are masked off in `extract_pad`
    // regardless).
    let mut bytes = vec![0u8; ohe.len() * stride + 8];
    // Stream scratch, reused across slots. Its own 8 tail bytes keep the u64
    // window loads in bounds; bytes past a slot's generated prefix are never
    // interpreted (windows are read only below the generated bit count), so
    // stale content from an earlier slot's extension is harmless.
    let init_blocks = total_bits.div_ceil(LAMBDA);
    let mut stream = vec![0u8; init_blocks * 16 + 8];
    for (i, l) in ohe.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let base = i * stride;
        hash::hash_bulk_into(
            l,
            group_id_base + i,
            init_blocks * LAMBDA,
            &mut stream[..init_blocks * 16],
        );
        let mut blocks = init_blocks;
        let mut max_win = blocks * LAMBDA / w; // whole windows generated so far
        let (mut member, mut win) = (0usize, 0usize);
        while member < b {
            if win == max_win {
                // Stream exhausted before the b-th accept: extend by one block.
                if stream.len() < (blocks + 1) * 16 + 8 {
                    stream.resize((blocks + 1) * 16 + 8, 0);
                }
                hash::hash_bulk_more(
                    l,
                    group_id_base + i,
                    blocks as u64,
                    &mut stream[blocks * 16..(blocks + 1) * 16],
                );
                blocks += 1;
                max_win = blocks * LAMBDA / w;
            }
            let v = extract_pad(&stream, 0, win, w);
            win += 1;
            if v < bound {
                pack_pad(&mut bytes, base, member, w, v);
                member += 1;
            }
        }
    }
    HashSlab { bytes, stride }
}

/// Pack member `j`'s accepted RAW pad (`v < 2^w`) into slot `i`'s slab region
/// at bits `[j·w .. (j+1)·w)` past `base = i·stride` — the exact layout
/// [`extract_pad`] reads back. Nibble alignment keeps the bit phase in
/// {0, 4}, so `v << phase` spans at most two bytes; the region is still zero
/// there (slots are packed once, members ascending), so OR-ing is a plain
/// store.
#[inline(always)]
fn pack_pad(slab: &mut [u8], base: usize, member_idx: usize, w: usize, v: u64) {
    let bit_off = member_idx * w;
    let byte = base + (bit_off >> 3);
    let phase = bit_off & 7;
    debug_assert!(phase + w <= 16 && v >> w == 0, "pack_pad writes ≤ 2 bytes");
    let sh = (v as u16) << phase;
    slab[byte] |= sh as u8;
    if phase + w > 8 {
        slab[byte + 1] |= (sh >> 8) as u8;
    }
}

/// Accumulate `Σ_i pad_i` and `Σ_i g_i·pad_i` per batch member into
/// `pad_sum_raw` / `readout_raw`, skipping slot `skip` entirely when given
/// (eval's active slot — always a whole-slot skip, never a per-member one).
///
/// Delayed reduction: pads are raw accepted `w`-bit windows (< 2^w, see
/// [`pad_width`] — rejection only thins the value range, so the bound is
/// unchanged) and weights are < p, so with p²·2^w < 2⁶³ the u64 sums `Σ pad_i`
/// (< p·2^w) and `Σ g_i·pad_i` (< p²·2^w, at most p terms) cannot overflow —
/// one `% p` per member at the fold replaces one per pad.
///
/// Dispatches to the NEON kernel on aarch64 when the modulus fits its u32
/// accumulators (see the gate below); the scalar kernel is both the portable
/// path and the fallback. The kernels produce bit-identical sums: each pad is
/// the same exact integer either way and no addition can overflow, so the
/// summation order is immaterial.
fn accumulate_pads(
    slab: &HashSlab,
    weights: &[u64],
    skip: Option<usize>,
    p_i: u64,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    let w = pad_width(p_i);
    assert!(
        (p_i as u128) * (p_i as u128) * (1u128 << w) < (1u128 << 63),
        "delayed-reduction accumulators would overflow for p = {p_i}"
    );
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // u32-accumulator bound, b-independent: readout_raw[j] sums at most
        // p terms (one per slot), each g·pad < p·2^w (pads are raw w-bit
        // slices), so the total is < p²·2^w; u32 is exact iff p²·2^w < 2³²,
        // i.e. p ≤ 1023 at w = 12. w ≤ 12 also keeps a 4-pad group inside
        // one u64 load. p ≥ 2 keeps the slab's exact region non-empty.
        let p128 = p_i as u128;
        if p_i >= 2 && p128 * p128 * (1u128 << w) < (1u128 << 32) {
            accumulate_pads_neon(slab, weights, skip, p_i, pad_sum_raw, readout_raw);
            return;
        }
    }
    accumulate_pads_scalar(slab, weights, skip, p_i, 0, pad_sum_raw, readout_raw);
}

/// Portable pad accumulation: slot-major walk, one [`extract_pad`] per
/// (slot, member). Covers members `j_lo..` only — the NEON kernel reuses it
/// for the ragged tail after its 4-member groups.
fn accumulate_pads_scalar(
    slab: &HashSlab,
    weights: &[u64],
    skip: Option<usize>,
    p_i: u64,
    j_lo: usize,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    let w = pad_width(p_i);
    for (i, &t) in weights.iter().enumerate() {
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
            let pad = extract_pad(&slab.bytes, base, j, w);
            *ps += pad;
            *ro += g * pad;
        }
    }
}

/// NEON pad accumulation: 4 members per u32 lane group, member-major so each
/// group's accumulators stay in registers across the whole slot walk.
///
/// Caller-guaranteed bounds (the dispatch gate): `2 ≤ p` and `p²·2^w < 2³²`.
/// Then
/// * the u32 lanes are exact — pad-sum lanes stay < p·2^w and readout lanes
///   < p²·2^w < 2³² across all ≤ p slots, so no lane ever wraps and one
///   widening into the u64 sums per group reproduces the scalar totals bit
///   for bit;
/// * w ≤ 12 with nibble alignment, so a group's four pads span
///   ≤ 4 + 4·12 = 52 bits of one unaligned u64 load (≤ 4 is the load's bit
///   phase, `shift0`).
///
/// Per lane this is the vector transcription of [`extract_pad`]: right-shift
/// the group word so the lane's pad starts at bit 0 (vshlq by a negative
/// count), mask to w bits — raw slices, no per-pad reduction.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn accumulate_pads_neon(
    slab: &HashSlab,
    weights: &[u64],
    skip: Option<usize>,
    p_i: u64,
    pad_sum_raw: &mut [u64],
    readout_raw: &mut [u64],
) {
    use core::arch::aarch64::*;

    let b = pad_sum_raw.len();
    let w = pad_width(p_i);
    debug_assert!((4..=12).contains(&w), "NEON gate admits 2 ≤ p ≤ 1023");
    // Per-slot weights g = weights[i] mod p, hoisted out of the
    // member-major walk (g < p ≤ 1023 fits u32 by the gate).
    let weights32: Vec<u32> = weights.iter().map(|&t| (t % p_i) as u32).collect();
    let full = b & !3; // members [0, full) in groups of 4; tail goes scalar.

    // SAFETY: NEON intrinsics — the cfg above pins target_arch + feature. The
    // u64 group loads start at `i·stride + (j0·lg_p >> 3)` with `j0 + 4 ≤ b`,
    // i.e. inside slot i's exact region (j0·lg_p < total_bits), which the
    // slab's 8 trailing bytes keep in bounds for an 8-byte read (see HashSlab).
    unsafe {
        let mask_v = vdupq_n_u32((1u32 << w) - 1);
        // Per-4-member-group constants: the group's byte offset into a slot
        // and the per-lane alignment shifts.
        let group_consts = |j0: usize| {
            let bit_off = j0 * w;
            let shift0 = (bit_off & 7) as i64;
            // vshlq with a negative count right-shifts: lane k's count drops
            // the bits below its pad, [shift0 + k·w ..][..w].
            let shifts = [0i64, 1, 2, 3].map(|k| -(shift0 + k * w as i64));
            (
                bit_off >> 3,
                vld1q_s64(shifts.as_ptr()),
                vld1q_s64(shifts.as_ptr().add(2)),
            )
        };
        // One group of 4 raw pads from a slot's u64 window (no reduction:
        // the fold's % p absorbs it — see extract_pad).
        let pads4 = |raw: u64, sh01, sh23| {
            let v = vdupq_n_u64(raw);
            vandq_u32(
                vcombine_u32(vmovn_u64(vshlq_u64(v, sh01)), vmovn_u64(vshlq_u64(v, sh23))),
                mask_v,
            )
        };
        let widen = |j0: usize, ps, ro, pad_sum_raw: &mut [u64], readout_raw: &mut [u64]| {
            let mut ps_arr = [0u32; 4];
            let mut ro_arr = [0u32; 4];
            vst1q_u32(ps_arr.as_mut_ptr(), ps);
            vst1q_u32(ro_arr.as_mut_ptr(), ro);
            for k in 0..4 {
                pad_sum_raw[j0 + k] += ps_arr[k] as u64;
                readout_raw[j0 + k] += ro_arr[k] as u64;
            }
        };

        // 8 members per slot pass: two independent accumulator pairs keep
        // both halves' dependency chains in flight on the NEON pipes and
        // halve the per-slot loop overhead (weights load, skip check, pointers).
        let full8 = b & !7;
        for j0 in (0..full8).step_by(8) {
            let (off_a, sh01_a, sh23_a) = group_consts(j0);
            let (off_b, sh01_b, sh23_b) = group_consts(j0 + 4);
            let mut ps_a = vdupq_n_u32(0);
            let mut ro_a = vdupq_n_u32(0);
            let mut ps_b = vdupq_n_u32(0);
            let mut ro_b = vdupq_n_u32(0);
            for (i, &g) in weights32.iter().enumerate() {
                if Some(i) == skip {
                    continue;
                }
                let base = slab.bytes.as_ptr().add(i * slab.stride);
                let raw_a = u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_a).cast()));
                let raw_b = u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_b).cast()));
                let pad_a = pads4(raw_a, sh01_a, sh23_a);
                let pad_b = pads4(raw_b, sh01_b, sh23_b);
                ps_a = vaddq_u32(ps_a, pad_a);
                ro_a = vmlaq_n_u32(ro_a, pad_a, g);
                ps_b = vaddq_u32(ps_b, pad_b);
                ro_b = vmlaq_n_u32(ro_b, pad_b, g);
            }
            widen(j0, ps_a, ro_a, pad_sum_raw, readout_raw);
            widen(j0 + 4, ps_b, ro_b, pad_sum_raw, readout_raw);
        }
        // Remaining full group of 4, if any.
        for j0 in (full8..full).step_by(4) {
            let (off_a, sh01_a, sh23_a) = group_consts(j0);
            let mut ps = vdupq_n_u32(0);
            let mut ro = vdupq_n_u32(0);
            for (i, &g) in weights32.iter().enumerate() {
                if Some(i) == skip {
                    continue;
                }
                let ptr = slab.bytes.as_ptr().add(i * slab.stride + off_a);
                let raw = u64::from_le_bytes(core::ptr::read_unaligned(ptr.cast::<[u8; 8]>()));
                let pad = pads4(raw, sh01_a, sh23_a);
                ps = vaddq_u32(ps, pad);
                ro = vmlaq_n_u32(ro, pad, g);
            }
            widen(j0, ps, ro, pad_sum_raw, readout_raw);
        }
    }
    // Ragged tail (b mod 4 members): scalar kernel, identical pads.
    if full < b {
        accumulate_pads_scalar(slab, weights, skip, p_i, full, pad_sum_raw, readout_raw);
    }
}

/// Member `j`'s RAW pad from slot `i`'s slab region — and, with `base = 0`,
/// window `j` of a slot's raw hash stream in [`slot_pads`]'s rejection scan:
/// bits `[j·w .. (j+1)·w)` past `base = i·stride`, w = [`pad_width`]. The
/// value is NOT reduced mod p — the accumulators run unreduced and the fold's
/// single `% p` per member reduces the sums (see [`accumulate_pads`]). Nibble
/// alignment keeps the bit phase in {0, 4}; the buffer's 8-byte tail keeps
/// the unconditional u64 load in bounds.
#[inline(always)]
fn extract_pad(slab: &[u8], base: usize, member_idx: usize, w: usize) -> u64 {
    debug_assert!(w <= 57, "extract_pad u64 load covers ≤ 57-bit slices");
    let bit_off = member_idx * w;
    let byte = base + (bit_off >> 3);
    let raw = u64::from_le_bytes(slab[byte..byte + 8].try_into().unwrap());
    (raw >> (bit_off & 7)) & ((1u64 << w) - 1)
}

// `Z_p`-share algebra. A label is its residue, so these operations are
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
        "Z_p modulus {p} exceeds the u64 fast-path bound"
    );
    (a * b) % p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngExt};

    fn rand_cf2_label<R: Rng>(rng: &mut R) -> Label {
        let coords: Vec<u64> = (0..crate::label::LAMBDA)
            .map(|_| rng.random_range(0..2u64))
            .collect();
        Label::from_coords(&coords, 2)
    }

    #[test]
    fn test_body_batch_kernel_round_trip_p3() {
        // Smallest case: p = 3, B = 4. h_p hot at index 1.
        let p_i = 3u64;
        let p = p_i as usize;
        let b = 4usize;
        let mut rng = rand::rng();

        // Construct h_p as the one-hot of value 1 (active position = index 1).
        let hot_idx = 1usize;
        let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
        let h_p_labels: Vec<Label> = (0..p)
            .map(|i| {
                if i == hot_idx {
                    // h_p[hot] = 1 ⇒ label = mask + Δ_2 (we just need the LSB to differ).
                    let mut coords = h_p_masks[i].to_coords();
                    coords[0] ^= 1;
                    Label::from_coords(&coords, 2)
                } else {
                    h_p_masks[i].clone()
                }
            })
            .collect();

        let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
        let weights: Vec<u64> = (0..p_i).collect();

        let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &weights, 0);
        let result_labels = body_batch_eval(
            p_i,
            hot_idx,
            &h_p_labels,
            &g_out.join_diffs,
            &b_batch,
            &weights,
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
    }

    /// Differential test for the SIMD dispatch: the auto kernel (NEON on
    /// aarch64 wherever the p³ < 2³² gate admits it, scalar everywhere else —
    /// every p below passes the gate) against the always-scalar kernel, end
    /// to end. Raw accumulator sums, garbler outputs (scaling residues + result
    /// masks) and eval outputs must be bit-identical, across power-of-two and
    /// odd moduli, ragged batch sizes, and a sweep of active positions.
    #[test]
    fn test_body_batch_simd_matches_scalar_differential() {
        let mut rng = rand::rng();
        let moduli: [u64; 14] = [2, 3, 5, 7, 17, 31, 127, 128, 129, 251, 257, 409, 1009, 1621];
        let batch_sizes: [usize; 12] = [1, 2, 3, 7, 8, 15, 16, 31, 64, 127, 128, 129];
        let gid = 17usize;
        for &p_i in &moduli {
            let p = p_i as usize;
            let weights: Vec<u64> = (0..p_i).collect();
            let mut hots = vec![0usize, 1 % p, p / 2, p - 1];
            hots.sort_unstable();
            hots.dedup();
            for &b in &batch_sizes {
                let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
                let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
                let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();

                // Garbler: full kernel (auto dispatch) vs the forced-scalar
                // reference rebuilt from the same deterministic slab.
                let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &weights, gid);
                let slab = slot_pads(&h_p_masks, gid, b, p_i, None);
                let (mut ps_auto, mut ro_auto) = (vec![0u64; b], vec![0u64; b]);
                accumulate_pads(&slab, &weights, None, p_i, &mut ps_auto, &mut ro_auto);
                let (mut ps_ref, mut ro_ref) = (vec![0u64; b], vec![0u64; b]);
                accumulate_pads_scalar(&slab, &weights, None, p_i, 0, &mut ps_ref, &mut ro_ref);
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
                                let mut coords = m.to_coords();
                                coords[0] ^= 1;
                                Label::from_coords(&coords, 2)
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
                        &weights,
                        gid,
                    );
                    let slab_l = slot_pads(&h_p_labels, gid, b, p_i, Some(hot));
                    let (mut ps_e, mut ro_e) = (vec![0u64; b], vec![0u64; b]);
                    accumulate_pads_scalar(
                        &slab_l,
                        &weights,
                        Some(hot),
                        p_i,
                        0,
                        &mut ps_e,
                        &mut ro_e,
                    );
                    let ev_ref = eval_outputs_from_sums(
                        p_i,
                        weights[hot] % p_i,
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
        let weights: Vec<u64> = (0..p_i).collect();

        for hot_idx in 0..p {
            let h_p_masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
            let h_p_labels: Vec<Label> = (0..p)
                .map(|i| {
                    if i == hot_idx {
                        let mut coords = h_p_masks[i].to_coords();
                        coords[0] ^= 1;
                        Label::from_coords(&coords, 2)
                    } else {
                        h_p_masks[i].clone()
                    }
                })
                .collect();

            let a_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();
            let b_batch: Vec<u64> = (0..b).map(|_| rng.random_range(0..p_i)).collect();

            let g_out = body_batch_garble(p_i, &h_p_masks, &a_batch, &b_batch, &weights, 0);
            let result_labels = body_batch_eval(
                p_i,
                hot_idx,
                &h_p_labels,
                &g_out.join_diffs,
                &b_batch,
                &weights,
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

    /// The width rule against a direct rational argmin of `w·2^w / acc` (with
    /// the smaller-width tie-break), plus the concrete bands on the 80-prime
    /// set and the moduli the differential test exercises.
    #[test]
    fn test_pad_width_rule() {
        let cost = |p: u64, w: usize| -> Option<(u64, u64)> {
            let acc = accept_bound(p, w);
            (acc != 0).then_some(((w as u64) << w, acc))
        };
        for p in (2u64..=421).chain([1009, 1621, 4093, 4099, 65521]) {
            let w = pad_width(p);
            let (n0, d0) = cost(p, w).expect("chosen width must admit acceptance");
            for cand in [4usize, 8, 12, 16] {
                let Some((n1, d1)) = cost(p, cand) else {
                    continue;
                };
                // Chosen width is no worse than any candidate, and strictly
                // better than any smaller one (ties break small).
                if cand < w {
                    assert!(n1 * d0 > n0 * d1, "p={p}: tie/loss to smaller w={cand}");
                } else {
                    assert!(n1 * d0 >= n0 * d1, "p={p}: w={w} beaten by w={cand}");
                }
            }
        }
        for &p in crate::crt::bigint::FIRST_80_PRIMES.iter() {
            let expect = match p {
                2..=13 => 4,
                17..=128 | 167..=251 => 8,
                _ => 12,
            };
            assert_eq!(pad_width(p), expect, "band mismatch at p={p}");
        }
        // Spot values: p = 2 rejects nothing at any width (the generic rule
        // picks 4); p = 167 is the exact 8-vs-12 tie (8·2^8/167 = 12·2^12/4008).
        assert_eq!((pad_width(2), accept_bound(2, 4)), (4, 16));
        assert_eq!((pad_width(167), accept_bound(167, 8)), (8, 167));
        assert_eq!((pad_width(131), accept_bound(131, 12)), (12, 4061));
        assert_eq!((pad_width(409), accept_bound(409, 12)), (12, 4090));
    }

    /// Every packed pad respects the acceptance bound (the rejection actually
    /// ran), and the eval-side `skip` leaves exactly that slot's region zero
    /// while reproducing the garbler's pads everywhere else.
    #[test]
    fn test_slot_pads_rejection_bound_and_skip() {
        let mut rng = rand::rng();
        for &p_i in &[3u64, 13, 131, 167, 409] {
            let p = p_i as usize;
            let b = 37usize;
            let w = pad_width(p_i);
            let bound = accept_bound(p_i, w);
            let masks: Vec<Label> = (0..p).map(|_| rand_cf2_label(&mut rng)).collect();
            let full = slot_pads(&masks, 99, b, p_i, None);
            let skip = p / 2;
            let skipped = slot_pads(&masks, 99, b, p_i, Some(skip));
            assert_eq!(full.stride, skipped.stride);
            for i in 0..p {
                for j in 0..b {
                    let v = extract_pad(&full.bytes, i * full.stride, j, w);
                    assert!(v < bound, "pad ≥ acceptance bound: p={p_i} slot={i} j={j}");
                    let vs = extract_pad(&skipped.bytes, i * skipped.stride, j, w);
                    assert_eq!(
                        vs,
                        if i == skip { 0 } else { v },
                        "skip divergence: p={p_i} slot={i} j={j}"
                    );
                }
            }
        }
    }

    /// Pins the expected-count ledger: exact `p·⌈b·w/λ⌉` when no window can
    /// reject (p = 2), else `⌈p·(b·w·2^w/(acc·λ) + 1/2)⌉`; join/program bits
    /// stay the communication-exact `b·lg p`.
    #[test]
    fn test_batch_cost_expected_hash_count() {
        for (p, b, expect) in [
            (2u64, 128usize, 8usize), // r = 0: deterministic 4 blocks × 2 slots
            (3, 128, 15),
            (17, 128, 146),
            (131, 128, 1652), // w = 12 band below the 8-vs-12 crossover
            (167, 128, 2132), // the tie prime, w = 8
            (409, 128, 5120),
            (409, 1, 243), // ragged batch: the ½-block term dominates
        ] {
            let c = batch_cost(p, b);
            assert_eq!(c.hash_count, expect, "hash_count at p={p} b={b}");
            assert_eq!(c.program_bits, b * hash::lg_modulus(p), "join at p={p}");
            assert_eq!(c.join_complexity, c.program_bits);
        }
    }
}
