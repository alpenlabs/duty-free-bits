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
/// — each slot draws windows until its `b`-th accept — so the ledger reports
/// a deterministic EXPECTED estimate, per band of [`pad_width`]'s rule
/// (`acc` = [`accept_bound`], `rem = 2^w − acc` the rejected values):
///
/// * `rem = 0` (p = 2): no window can reject — the exact `p·⌈b·w/λ⌉`.
/// * w = 16: rejects are so rare (≤ 0.54%) that a slot pays at most one
///   spare block beyond its `⌈b·w/λ⌉` floor, with probability
///   `P(any reject) = 1 − (acc/2^w)^b ≈ b·rem/(b·rem + acc)` (a one-pole
///   Padé form that stays in integers) — so `p·(⌈b·w/λ⌉ + b·rem/(b·rem +
///   acc))`, rounded to nearest.
/// * w = 4 (p ≤ 13, reject rates up to 3/16): multiple spare blocks are in
///   play, so the negative-binomial mean bits `b·w·2^w/acc` over λ plus half
///   a block for whole-block draws.
///
/// `bench_axb_hashcounts` cross-checks the measured garbler count against
/// this to ±3%; on the 80-prime set the estimate sits within ~0.2% of the
/// true expectation (the w = 4 terms are ~0.04% of the total).
fn batch_cost(p_i: u64, b: usize) -> Cost {
    let join_bits = b * hash::lg_modulus(p_i);
    let w = pad_width(p_i);
    let acc = accept_bound(p_i, w);
    let (bw, lambda) = ((b * w) as u64, LAMBDA as u64);
    let rem = (1u64 << w) - acc;
    let hash_count = if rem == 0 {
        p_i * bw.div_ceil(lambda)
    } else if w == 4 {
        // p·(b·w·2^w/(acc·λ) + 1/2), rounded up in exact integer arithmetic.
        (p_i * (2 * bw * (1u64 << w) + acc * lambda)).div_ceil(2 * acc * lambda)
    } else {
        // p·(⌈b·w/λ⌉ + n/(n + acc)) with n = b·rem, rounded to nearest
        // (all terms ≤ ~2^40 — no overflow near u64).
        let n = b as u64 * rem;
        let d = n + acc;
        (p_i * (bw.div_ceil(lambda) * d + n) + d / 2) / d
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

/// Bits of hash output sliced per sampling attempt: `w = 16` for every prime
/// p ≥ 17, `w = 4` for the tiny primes p ≤ 13.
///
/// Rejection against [`accept_bound`] makes the pads EXACTLY uniform over
/// `Z_p` — see [`slot_pads`] — where reducing a bare `w`-bit slice mod `p` is
/// biased by the rejection rate `r(w) = (2^w mod p) / 2^w` (up to ~49% at
/// p = 131 for the former nibble-aligned `w = 8`). Any nibble-aligned width
/// with `2^w ≥ p` buys that exactness; the width trades hash bits per attempt
/// against `r(w)` — and, because a rejected window costs a data-dependent
/// scalar fixup, against the reject VOLUME the body must patch. A 16-bit
/// window for every p ≥ 17 makes both terms small at once: r ≤ 409/2^16 —
/// under 0.54% everywhere on the 80-prime set (the worst is 348/2^16 at
/// p = 379), so fixups are ~0.25% of windows — and the slab cells become
/// whole u16 lanes, so the reject scan is a bare vector compare
/// ([`w16_scan_chunk_neon`], no gather) and an accumulate group is exactly
/// one u64 load. Narrower windows were measured and rejected: w = 12
/// (rejects up to 8.9%) spends more time patching rejects than it saves in
/// hash bits — see the commit history. The six primes ≤ 13 keep the 4-bit
/// window (a quarter of the hash bits; their 41 slots per batch and reject
/// handling are noise). `test_pad_width_pins` pins the table.
///
/// Both widths are nibble multiples, so window extraction and patching stay
/// load + shift + mask (bit phase 0 or 4 — always 0 at w = 16) with no
/// per-pad reduction ([`extract_pad`] / [`patch_pad`]). The scaling
/// residues — the actual communication — remain `lg|p_i|`-bit.
fn pad_width(p_i: u64) -> usize {
    // 2^w ≥ p keeps accept_bound > 0 — below, some window can accept.
    assert!(
        (2..=1 << 16).contains(&p_i),
        "pad_width covers 2 ≤ p ≤ 2^16 (the concrete primes are ≤ 409)"
    );
    if p_i <= 13 { 4 } else { 16 }
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
/// Sampling is by IN-PLACE PROBING, so the bulk of the work is the
/// pre-rejection pipeline unchanged: the stream's first `b` windows land in
/// the slab verbatim (for whole-block batches — every driver batch — the CCRH
/// writes the slab region directly), one reject scan compares the installed
/// windows against [`accept_bound`] ([`scan_chunk`]: NEON across the
/// w = 16 band, the scalar mirror elsewhere), and each rejected member —
/// ~0.25% of windows at [`pad_width`]'s widths — is patched, in increasing
/// member order, with the next accepted window drawn from the slot's
/// [`SpareCursor`]. Member `j`'s pad is thus the first accepted window among
/// those this deterministic rule assigns it — exactly uniform mod `p` — and
/// the stream is consumed through the (b + #rejects)-th window, the same
/// count as sequential rejection sampling, with unbounded retries (geometric
/// tail, no fallback). The stream extends one block at a time via
/// [`hash::hash_bulk_more`] under the slot's one nonce: the CTR block
/// counter, not the nonce, sequences a stream, so the [`crate::affine`]
/// nonce layout is untouched by rejections. Accepted windows stay RAW (not
/// reduced): the fold's single `% p` per member absorbs the reduction,
/// exactly as before (`Σ vᵢ ≡ Σ (vᵢ mod p)`).
///
/// Garbler and evaluator call this identically (on masks / labels): at every
/// off slot the two streams agree bit for bit, so the reject pattern — and
/// hence every pad — lines up. At the active slot the evaluator's stream
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
    let init_blocks = total_bits.div_ceil(LAMBDA);
    // Whole-block batches (b·w ≡ 0 mod λ, i.e. every b = 128 driver batch)
    // have no spare bits in the initial blocks: the CCRH writes the slab
    // region directly and spares start at a fresh block. A ragged batch
    // detours through the scratch so the last block's tail windows survive
    // as spares.
    let whole = total_bits.is_multiple_of(LAMBDA);
    // One allocation for the whole batch; intra-slot padding, a skipped slot
    // and the 8-byte tail stay zero (and are masked off in `extract_pad`
    // regardless).
    let mut bytes = vec![0u8; ohe.len() * stride + 8];
    // Spare-window scratch, reused across slots (see SpareCursor); its 8 tail
    // bytes keep the u64 window loads in bounds, and stale bytes past a
    // slot's generated prefix are never interpreted (each block is written
    // before any window in it is read).
    let mut scratch: Vec<u8> = Vec::new();
    // A power-of-two 2^w (p = 2's w = 4, and power-of-two test moduli at
    // w = 16) accepts every window: nothing can reject, so the scan pass is
    // skipped outright.
    let rejectable = bound < 1u64 << w;
    if whole {
        // Pass 1: hash every slot's stream head straight into the slab.
        for (i, l) in ohe.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            hash::hash_bulk_into(
                l,
                group_id_base + i,
                total_bits,
                &mut bytes[i * stride..i * stride + exact_len],
            );
        }
        // Pass 2: one flat sweep of 64-window reject scans over every slot
        // (a skipped slot's zeroed region yields a zero bitmap — scanning it
        // is cheaper than a per-slot branch), storing per-chunk reject
        // bitmaps without a single branch or vector→scalar crossing; pass 3
        // walks the bitmaps — almost all zero at the concrete reject rates —
        // and patches. Fusing scan into the hash loop, or testing each
        // chunk's bitmap as it is produced, was measured 2–3x slower: the
        // short per-chunk dependency chain (compare tree, lane extraction,
        // branch) serializes against its neighbors.
        if rejectable {
            // whole ⟹ λ | b·w ⟹ 8 | b, so there is no sub-group tail.
            debug_assert!(b.is_multiple_of(8), "whole-block batches scan cleanly");
            let maps_per_slot = b.div_ceil(64);
            let mut bitmaps = vec![0u64; ohe.len() * maps_per_slot];
            for (i, slot_maps) in bitmaps.chunks_exact_mut(maps_per_slot).enumerate() {
                let base = i * stride;
                for (m, out) in slot_maps.iter_mut().enumerate() {
                    let win0 = m * 64;
                    scan_chunk_into(&bytes, base, win0, (b - win0).min(64), w, bound, out);
                }
            }
            // Pass 3: patch. Bitmaps are walked in (slot, member) order, one
            // cursor per dirty slot, so the spare assignment is deterministic
            // and member-ordered.
            for (i, l) in ohe.iter().enumerate() {
                if Some(i) == skip {
                    continue;
                }
                let maps = &bitmaps[i * maps_per_slot..(i + 1) * maps_per_slot];
                if maps.iter().all(|&m| m == 0) {
                    continue;
                }
                let mut cursor = SpareCursor::new(b, init_blocks, whole, w);
                for (m, &bm0) in maps.iter().enumerate() {
                    let mut bm = bm0;
                    while bm != 0 {
                        let t = m * 64 + bm.trailing_zeros() as usize;
                        let v = cursor.next_accepted(l, group_id_base + i, w, bound, &mut scratch);
                        patch_pad(&mut bytes, i * stride, t, w, v);
                        bm &= bm - 1;
                    }
                }
            }
        }
    } else {
        // Ragged batch: the stream head detours through the scratch (so the
        // last block's tail windows survive as spares), which the next slot
        // reuses — hash, install, and fix one slot at a time.
        for (i, l) in ohe.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            if scratch.len() < init_blocks * 16 + 8 {
                scratch.resize(init_blocks * 16 + 8, 0);
            }
            hash::hash_bulk_into(
                l,
                group_id_base + i,
                init_blocks * LAMBDA,
                &mut scratch[..init_blocks * 16],
            );
            bytes[i * stride..i * stride + exact_len].copy_from_slice(&scratch[..exact_len]);
            if rejectable {
                let mut cursor = SpareCursor::new(b, init_blocks, whole, w);
                fix_slot_rejects(
                    &mut bytes[i * stride..],
                    l,
                    group_id_base + i,
                    b,
                    p_i,
                    &mut cursor,
                    &mut scratch,
                );
            }
        }
    }
    HashSlab { bytes, stride }
}

/// Scan one slot's installed windows (`region` starts at the slot's slab
/// base) and patch every member the acceptance bound rejects — see
/// [`slot_pads`]. The reject scan runs in chunks of up to 64 windows, each
/// returning a u64 reject bitmap (bit `t` = window `win0 + t`): the common
/// all-accept chunk costs one vector pass and one zero test; set bits are
/// walked in increasing member order, each patched with the cursor's next
/// accepted spare.
fn fix_slot_rejects(
    region: &mut [u8],
    l: &Label,
    group_id: usize,
    b: usize,
    p_i: u64,
    cursor: &mut SpareCursor,
    scratch: &mut Vec<u8>,
) {
    let w = pad_width(p_i);
    let bound = accept_bound(p_i, w);
    let full = b & !7;
    let mut win0 = 0usize;
    while win0 < full {
        let windows = (full - win0).min(64);
        let mut bm = scan_chunk(region, 0, win0, windows, w, bound);
        while bm != 0 {
            let t = bm.trailing_zeros() as usize;
            let v = cursor.next_accepted(l, group_id, w, bound, scratch);
            patch_pad(region, 0, win0 + t, w, v);
            bm &= bm - 1;
        }
        win0 += windows;
    }
    for j in full..b {
        if extract_pad(region, 0, j, w) >= bound {
            let v = cursor.next_accepted(l, group_id, w, bound, scratch);
            patch_pad(region, 0, j, w, v);
        }
    }
}

/// Deterministic spare-window cursor for one slot's fixups: probes windows
/// `b, b+1, …` of the slot's stream — the windows past the `b` installed
/// ones — returning the next accepted value per call. The probe order is a
/// pure function of the shared stream, so garbler and evaluator hand the same
/// spare to the same rejected member. `scratch` holds the stream's blocks
/// past those installed in the slab, growing a block at a time.
struct SpareCursor {
    /// Next absolute window index to probe.
    win: usize,
    /// Absolute window index sitting at scratch bit 0.
    win_base: usize,
    /// Absolute block index sitting at scratch byte 0.
    first_block: usize,
    /// Absolute blocks generated so far (slab-installed + scratch).
    blocks: usize,
}

impl SpareCursor {
    fn new(b: usize, init_blocks: usize, whole: bool, w: usize) -> Self {
        // Whole-block batches: the slab holds all init_blocks, scratch starts
        // empty at block init_blocks = window b (b·w = init_blocks·λ). Ragged
        // batches: scratch already holds blocks 0.., so spares index from 0.
        let (win_base, first_block) = if whole { (b, init_blocks) } else { (0, 0) };
        debug_assert_eq!(first_block * LAMBDA, win_base * w, "cursor phase");
        SpareCursor {
            win: b,
            win_base,
            first_block,
            blocks: init_blocks,
        }
    }

    fn next_accepted(
        &mut self,
        l: &Label,
        group_id: usize,
        w: usize,
        bound: u64,
        scratch: &mut Vec<u8>,
    ) -> u64 {
        loop {
            if (self.win + 1) * w > self.blocks * LAMBDA {
                // Spares exhausted: continue the slot's CTR stream one block.
                let have = self.blocks - self.first_block;
                if scratch.len() < (have + 1) * 16 + 8 {
                    scratch.resize((have + 1) * 16 + 8, 0);
                }
                hash::hash_bulk_more(
                    l,
                    group_id,
                    self.blocks as u64,
                    &mut scratch[have * 16..(have + 1) * 16],
                );
                self.blocks += 1;
            }
            let v = extract_pad(scratch, 0, self.win - self.win_base, w);
            self.win += 1;
            if v < bound {
                return v;
            }
        }
    }
}

/// Reject bitmap of `windows` consecutive windows (a multiple of 8, ≤ 64)
/// starting at window `first_win` (also a multiple of 8) of a slot's slab
/// region, stored into `out`: bit `t` is set iff window `first_win + t` is
/// ≥ `bound`. Zero — the overwhelmingly common chunk — means nothing to
/// patch. NEON across the w = 16 band (every prime ≥ 17), where the bitmap
/// is written by a VECTOR store (the caller reads it back well after, so no
/// crossing and no forwarding stall sits on the scan's critical path); the
/// scalar mirror elsewhere — decision-identical by
/// `test_w16_scan_chunk_neon_matches_scalar`.
#[inline(always)]
fn scan_chunk_into(
    slab: &[u8],
    base: usize,
    first_win: usize,
    windows: usize,
    w: usize,
    bound: u64,
    out: &mut u64,
) {
    debug_assert!(
        first_win.is_multiple_of(8) && windows.is_multiple_of(8) && (8..=64).contains(&windows),
        "scan_chunk takes whole 8-window groups"
    );
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if w == 16 {
        w16_scan_chunk_neon(slab, base + first_win * 2, windows, bound, out);
        return;
    }
    *out = scan_chunk_scalar(slab, base, first_win, windows, w, bound);
}

/// [`scan_chunk_into`] returning the bitmap — for the ragged path and tests,
/// where the extra store/reload is off the hot path.
#[inline(always)]
fn scan_chunk(
    slab: &[u8],
    base: usize,
    first_win: usize,
    windows: usize,
    w: usize,
    bound: u64,
) -> u64 {
    let mut bm = 0u64;
    scan_chunk_into(slab, base, first_win, windows, w, bound, &mut bm);
    bm
}

/// Portable chunk scan: one [`extract_pad`] + compare per window.
fn scan_chunk_scalar(
    slab: &[u8],
    base: usize,
    first_win: usize,
    windows: usize,
    w: usize,
    bound: u64,
) -> u64 {
    let mut bm = 0u64;
    for t in 0..windows {
        if extract_pad(slab, base, first_win + t, w) >= bound {
            bm |= 1u64 << t;
        }
    }
    bm
}

/// NEON chunk scan for the 16-bit band: the slab cells ARE u16 lanes, so a
/// vector of 8 windows is one plain load — no gather. Each vector is compared
/// against the acceptance bound; the lane masks are narrowed to bytes, ANDed
/// with per-lane bit weights (window `k` of a 16-lane pair owns bit `k mod
/// 8`), and folded by a pairwise-add tree into one u8x16 whose low 8 bytes
/// are the chunk's reject bitmap, vector-stored into `out`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn w16_scan_chunk_neon(slab: &[u8], byte_off: usize, windows: usize, bound: u64, out: &mut u64) {
    use core::arch::aarch64::*;
    // Window k of a 16-lane pair contributes bit k mod 8 of its group byte.
    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    debug_assert!(
        byte_off + 2 * windows <= slab.len(),
        "chunk load out of bounds"
    );
    debug_assert!(
        bound < 1 << 16,
        "w = 16 acceptance bound (2^16 never scans)"
    );
    // SAFETY: NEON intrinsics — the cfg pins target_arch + feature. Every
    // 16-byte load covers 8 whole 16-bit cells of the slot's exact region
    // (the debug_assert above pins the last load in bounds); the 8-byte
    // store writes the caller's `out`.
    unsafe {
        let weights = vld1q_u8(WEIGHTS.as_ptr());
        let boundv = vdupq_n_u16(bound as u16);
        let zero = vdup_n_u8(0);
        // Reject mask bytes of vector `v` (8 windows), or all-accept for a
        // vector past the chunk's end.
        let mask8 = |v: usize| {
            if v * 8 < windows {
                let cells = vld1q_u16(slab.as_ptr().add(byte_off + 16 * v).cast());
                vmovn_u16(vcgeq_u16(cells, boundv))
            } else {
                zero
            }
        };
        // Four 16-lane pairs of weighted masks; a 3-level pairwise-add tree
        // sums each 8-lane group's distinct bit weights into its group byte:
        // low 8 bytes of `s` = the 64-window bitmap, little-endian.
        let t = |v: usize| vandq_u8(vcombine_u8(mask8(2 * v), mask8(2 * v + 1)), weights);
        let p = vpaddq_u8(t(0), t(1));
        let q = vpaddq_u8(t(2), t(3));
        let r = vpaddq_u8(p, q);
        let s = vpaddq_u8(r, r);
        vst1_u8((out as *mut u64).cast::<u8>(), vget_low_u8(s));
    }
}

/// Patch member `j`'s slab position with an accepted spare (`v < 2^w`),
/// replacing the rejected window at bits `[j·w .. (j+1)·w)` past
/// `base = i·stride`. Nibble alignment keeps the bit phase in {0, 4}, so the
/// read-modify-write spans at most two bytes; bits outside the window are
/// written back unchanged.
#[inline(always)]
fn patch_pad(slab: &mut [u8], base: usize, member_idx: usize, w: usize, v: u64) {
    let bit_off = member_idx * w;
    let byte = base + (bit_off >> 3);
    let phase = bit_off & 7;
    debug_assert!(phase + w <= 16 && v >> w == 0, "patch_pad writes ≤ 2 bytes");
    let mask = (((1u32 << w) - 1) << phase) as u16;
    let cur = u16::from_le_bytes([slab[byte], slab[byte + 1]]);
    let le = ((cur & !mask) | ((v as u16) << phase)).to_le_bytes();
    slab[byte] = le[0];
    slab[byte + 1] = le[1];
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
/// Dispatches to the NEON kernel on aarch64 when one `g·pad` term fits its
/// u32 accumulator lanes (the kernel spills the lane sums into the u64
/// totals often enough that this is the only bound — see the spill note
/// there); the scalar kernel is both the portable path and the fallback. The
/// kernels produce bit-identical sums: each pad is the same exact integer
/// either way and no addition can overflow, so the summation order is
/// immaterial.
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
        // One readout term g·pad ≤ (p−1)·(2^w − 1) must fit a u32 lane (so
        // the kernel's spill window is ≥ 1 slot); p ≥ 2 keeps the slab's
        // exact region non-empty. Every production modulus passes.
        if p_i >= 2 && (p_i - 1) * ((1u64 << w) - 1) <= u32::MAX as u64 {
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
/// group's accumulators stay in registers across the slot walk.
///
/// The u32 lanes are kept exact by SPILLING: the slot walk runs in spans of
/// `spill = ⌊(2³² − 1) / ((p−1)·(2^w − 1))⌋` slots, each span's lane sums
/// widened into the u64 totals before the next begins, so a readout lane
/// (≤ spill terms of `g·pad ≤ (p−1)·(2^w − 1)`) never wraps — the dispatch
/// gate guarantees `spill ≥ 1`, and pad-sum lanes are bounded tighter. One
/// span covers every p ≤ 255 at w = 16 (and all smaller widths up to
/// p = 1023); the larger 16-bit-band primes spill 2–3 times per walk, which
/// widening is too rare to see. Bit-identical to the scalar kernel either
/// way: every partial sum is an exact integer, so the widening points are
/// immaterial.
///
/// Group geometry: nibble alignment keeps a group's bit phase `shift0` in
/// {0, 4} (always 0 at w = 16), so four pads span `shift0 + 4w ≤ 64` bits of
/// one unaligned u64 load. Per lane this is the vector transcription of
/// [`extract_pad`]: right-shift the group word so the lane's pad starts at
/// bit 0 (vshlq by a negative count), mask to w bits — raw slices, no
/// per-pad reduction.
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
    debug_assert!(w == 4 || w == 16, "pad_width's widths");
    // Slots per u32 accumulation span (see the doc comment). The dispatch
    // gate makes the divisor ≤ u32::MAX, so spill ≥ 1.
    let spill = (u32::MAX as u64 / ((p_i - 1).max(1) * ((1u64 << w) - 1))).max(1) as usize;
    // Per-slot weights g = weights[i] mod p, hoisted out of the
    // member-major walk (g < p ≤ 2^16 fits u32 by the gate).
    let weights32: Vec<u32> = weights.iter().map(|&t| (t % p_i) as u32).collect();
    let full = b & !3; // members [0, full) in groups of 4; tail goes scalar.

    // SAFETY: NEON intrinsics — the cfg above pins target_arch + feature. The
    // u64 group loads start at `i·stride + (j0·w >> 3)` with `j0 + 4 ≤ b`,
    // i.e. inside slot i's exact region (j0·w < total_bits), which the
    // slab's 8 trailing bytes keep in bounds for an 8-byte read (see HashSlab).
    unsafe {
        let mask_v = vdupq_n_u32(((1u64 << w) - 1) as u32);
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
        // the fold's % p absorbs it — see extract_pad). At w = 16 the four
        // pads ARE the load's four aligned u16 lanes, so extraction is a
        // single widening move — the dominant band skips the shift/mask
        // surgery entirely (LLVM unswitches the slot walk on the
        // loop-invariant width, so the branch costs nothing there).
        let w16 = w == 16;
        let pads4 = |raw: u64, sh01, sh23| {
            if w16 {
                return vmovl_u16(vcreate_u16(raw));
            }
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

        // 16 members per slot pass: four independent accumulator pairs keep
        // the dependency chains in flight on the NEON pipes, and — since the
        // walk streams the whole slab once per pass — quarter the slab
        // traffic and per-slot loop overhead (weights load, skip check,
        // pointers) relative to a 4-member pass. At b = 128 with 16-bit
        // cells the slab re-streams 8× per batch, which is what keeps the
        // member-major layout memory-neutral against the old 8-bit-era
        // slab. The slot walk runs one u32 span (≤ `spill` slots) at a time.
        let full16 = b & !15;
        for j0 in (0..full16).step_by(16) {
            let (off_a, sh01_a, sh23_a) = group_consts(j0);
            let (off_b, sh01_b, sh23_b) = group_consts(j0 + 4);
            let (off_c, sh01_c, sh23_c) = group_consts(j0 + 8);
            let (off_d, sh01_d, sh23_d) = group_consts(j0 + 12);
            for (span, span_weights) in weights32.chunks(spill).enumerate() {
                let i0 = span * spill;
                let mut ps_a = vdupq_n_u32(0);
                let mut ro_a = vdupq_n_u32(0);
                let mut ps_b = vdupq_n_u32(0);
                let mut ro_b = vdupq_n_u32(0);
                let mut ps_c = vdupq_n_u32(0);
                let mut ro_c = vdupq_n_u32(0);
                let mut ps_d = vdupq_n_u32(0);
                let mut ro_d = vdupq_n_u32(0);
                for (di, &g) in span_weights.iter().enumerate() {
                    if Some(i0 + di) == skip {
                        continue;
                    }
                    let base = slab.bytes.as_ptr().add((i0 + di) * slab.stride);
                    let raw_a =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_a).cast()));
                    let raw_b =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_b).cast()));
                    let raw_c =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_c).cast()));
                    let raw_d =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_d).cast()));
                    let pad_a = pads4(raw_a, sh01_a, sh23_a);
                    let pad_b = pads4(raw_b, sh01_b, sh23_b);
                    let pad_c = pads4(raw_c, sh01_c, sh23_c);
                    let pad_d = pads4(raw_d, sh01_d, sh23_d);
                    ps_a = vaddq_u32(ps_a, pad_a);
                    ro_a = vmlaq_n_u32(ro_a, pad_a, g);
                    ps_b = vaddq_u32(ps_b, pad_b);
                    ro_b = vmlaq_n_u32(ro_b, pad_b, g);
                    ps_c = vaddq_u32(ps_c, pad_c);
                    ro_c = vmlaq_n_u32(ro_c, pad_c, g);
                    ps_d = vaddq_u32(ps_d, pad_d);
                    ro_d = vmlaq_n_u32(ro_d, pad_d, g);
                }
                widen(j0, ps_a, ro_a, pad_sum_raw, readout_raw);
                widen(j0 + 4, ps_b, ro_b, pad_sum_raw, readout_raw);
                widen(j0 + 8, ps_c, ro_c, pad_sum_raw, readout_raw);
                widen(j0 + 12, ps_d, ro_d, pad_sum_raw, readout_raw);
            }
        }
        // Remaining group of 8, if any.
        let full8 = b & !7;
        for j0 in (full16..full8).step_by(8) {
            let (off_a, sh01_a, sh23_a) = group_consts(j0);
            let (off_b, sh01_b, sh23_b) = group_consts(j0 + 4);
            for (span, span_weights) in weights32.chunks(spill).enumerate() {
                let i0 = span * spill;
                let mut ps_a = vdupq_n_u32(0);
                let mut ro_a = vdupq_n_u32(0);
                let mut ps_b = vdupq_n_u32(0);
                let mut ro_b = vdupq_n_u32(0);
                for (di, &g) in span_weights.iter().enumerate() {
                    if Some(i0 + di) == skip {
                        continue;
                    }
                    let base = slab.bytes.as_ptr().add((i0 + di) * slab.stride);
                    let raw_a =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_a).cast()));
                    let raw_b =
                        u64::from_le_bytes(core::ptr::read_unaligned(base.add(off_b).cast()));
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
        }
        // Remaining full group of 4, if any.
        for j0 in (full8..full).step_by(4) {
            let (off_a, sh01_a, sh23_a) = group_consts(j0);
            for (span, span_weights) in weights32.chunks(spill).enumerate() {
                let i0 = span * spill;
                let mut ps = vdupq_n_u32(0);
                let mut ro = vdupq_n_u32(0);
                for (di, &g) in span_weights.iter().enumerate() {
                    if Some(i0 + di) == skip {
                        continue;
                    }
                    let ptr = slab.bytes.as_ptr().add((i0 + di) * slab.stride + off_a);
                    let raw = u64::from_le_bytes(core::ptr::read_unaligned(ptr.cast::<[u8; 8]>()));
                    let pad = pads4(raw, sh01_a, sh23_a);
                    ps = vaddq_u32(ps, pad);
                    ro = vmlaq_n_u32(ro, pad, g);
                }
                widen(j0, ps, ro, pad_sum_raw, readout_raw);
            }
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

    /// The width table (w = 4 below 17, w = 16 above), the acceptance-bound
    /// arithmetic at the band edges, and the reject-rate bound the width doc
    /// claims for the 80-prime set.
    #[test]
    fn test_pad_width_pins() {
        for p in (2u64..=421).chain([1009, 1621, 4093, 65521]) {
            let w = pad_width(p);
            assert_eq!(w, if p <= 13 { 4 } else { 16 }, "width table at p={p}");
            let acc = accept_bound(p, w);
            // Acceptance region is the largest multiple of p below 2^w.
            assert!(
                acc > 0 && acc.is_multiple_of(p) && (1u64 << w) - acc < p,
                "p={p}"
            );
        }
        // Spot values: p = 2 rejects nothing; the mid primes that motivated
        // rejection sit in the 16-bit band with <= 0.54% reject rates.
        assert_eq!(accept_bound(2, 4), 16);
        assert_eq!(accept_bound(13, 4), 13);
        assert_eq!(accept_bound(17, 16), 65535);
        assert_eq!(accept_bound(131, 16), 65500);
        assert_eq!(accept_bound(409, 16), 65440);
        // Worst reject rate on the 80-prime set: 348/2^16 (~0.53%) at p = 379.
        let worst = crate::crt::bigint::FIRST_80_PRIMES
            .iter()
            .filter(|&&p| p >= 17)
            .map(|&p| ((1u64 << 16) - accept_bound(p, 16), p))
            .max()
            .unwrap();
        assert_eq!(worst, (348, 379), "reject-rate bound in the pad_width doc");
    }

    /// The NEON chunk scan against the scalar mirror: identical reject
    /// bitmaps on random window data across chunk sizes and bounds
    /// (including the always-reject bound 0 and the near-boundary values;
    /// bound 2^16 never reaches the scan -- slot_pads skips it).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn test_w16_scan_chunk_neon_matches_scalar() {
        let mut rng = rand::rng();
        let w = 16usize;
        for &bound in &[0u64, 1, 32768, 65207, 65440, 65500, 65534, 65535] {
            for &windows in &[8usize, 16, 40, 64] {
                for _ in 0..20 {
                    // The chunk's cells plus the 16-byte load tail.
                    let bytes: Vec<u8> = (0..64 * 2 + 16).map(|_| rng.random::<u8>()).collect();
                    let mut got = 0u64;
                    w16_scan_chunk_neon(&bytes, 0, windows, bound, &mut got);
                    assert_eq!(
                        got,
                        scan_chunk_scalar(&bytes, 0, 0, windows, w, bound),
                        "bound={bound} windows={windows}"
                    );
                }
            }
        }
    }

    /// Every installed pad respects the acceptance bound (the reject scan and
    /// fixups actually ran), and the eval-side `skip` leaves exactly that
    /// slot's region zero while reproducing the garbler's pads everywhere
    /// else. `b = 128` exercises the whole-block direct-to-slab path (and the
    /// NEON scan on aarch64); `b = 37` the ragged scratch detour and the
    /// scalar group tail.
    #[test]
    fn test_slot_pads_rejection_bound_and_skip() {
        let mut rng = rand::rng();
        for &p_i in &[3u64, 13, 131, 373, 409] {
            for b in [37usize, 128] {
                let p = p_i as usize;
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
    }

    /// Pins the expected-count ledger: exact `p·⌈b·w/λ⌉` when no window can
    /// reject (p = 2), else `⌈p·(b·w·2^w/(acc·λ) + 1/2)⌉`; join/program bits
    /// stay the communication-exact `b·lg p`.
    #[test]
    fn test_batch_cost_expected_hash_count() {
        for (p, b, expect) in [
            (2u64, 128usize, 8usize), // rem = 0: deterministic 4 blocks x 2 slots
            (3, 128, 15),
            (13, 128, 71),    // w = 4 band: the negative-binomial + half-block model
            (17, 128, 272),   // 16-bit band: 16-block floor + P(any reject)
            (131, 128, 2105), // the worst-biased prime under the old slicing
            (251, 128, 4028),
            (257, 128, 4113),
            (331, 128, 5426),
            (409, 128, 6609),
            (409, 1, 410), // ragged batch: one block per slot + the reject term
        ] {
            let c = batch_cost(p, b);
            assert_eq!(c.hash_count, expect, "hash_count at p={p} b={b}");
            assert_eq!(c.program_bits, b * hash::lg_modulus(p), "join at p={p}");
            assert_eq!(c.join_complexity, c.program_bits);
        }
    }
}
