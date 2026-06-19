//! Garbler: build masks, encode program.
//!
//! Structure mirrors `Exec`: seed constants + inputs, then a worklist fires
//! gates as mask inputs become available. Mask rules per gate:
//!
//! * `Add`/`Sub`: bidirectional — any two of (in0, in1, out) determine the third.
//! * `Mul(s, x)`: the output mask is `s · X_x`. When `s = 0` this is `0`
//!   regardless of `X_x`, so the gate can fire without its input being masked (required to break circularity in the last iteration of `word_to_hot_with_bits`)
//! * `Mod2k` / `Div2k`: forward only (non-invertible at the mask level).
//! * `Switch`: bidirectional (ctrl+data → out, ctrl+out → data) using
//!   `Y = H(X_ctrl, gid) + X_data`. Garbler does not branch on ctrl value —
//!   masks propagate unconditionally.
//! * `Join`: does not propagate masks (both sides are independently determined).
//! * `SameWire`: symmetric copy.
//!
//! Δ is the global Free-XOR offset: sampled uniformly at random and kept secret
//! from the evaluator.
//! A final pass over gates in creation order emits the join diffs (the only
//! switch-system communication — switches reveal nothing) and appends the
//! declared output masks.
//!
//! Two entry points share one engine (`seed_masks` → propagation →
//! `emit_program`):
//! * [`garble`] — worklist propagation to fixpoint.
//! * [`garble_recorded`] — same, additionally recording the firing schedule.
//!   The schedule depends only on the system's *structure* (never on mask
//!   values), so the gate-for-gate identical prime phases record it once and
//!   subsequent phases of the same shape compile and run it without a
//!   worklist (see the `arena` module).

use super::program::Program;
use crate::exec::{JournalEntry, Worklist};
use crate::hash;
use crate::label::{self, Label, NcfLabel};
use crate::system::System;
use crate::types::{Gate, GateId, Wire};

/// Lazy cache for per-group bulk-hash outputs. `bulk_cache[group_idx]` is
/// computed on first use and reused for every member of that group.
type BulkCache = Vec<Option<Vec<u8>>>;

/// Resolve the H value for a switch gate, packing across group members
/// when the gate is registered in a [`SwitchGroup`].
fn switch_hash(
    system: &System,
    gid: GateId,
    ctrl_mask: &Label,
    bulk_cache: &mut BulkCache,
    nonce_base: u64,
) -> Label {
    if let Some((group_idx, member_idx)) = system.gate_group(gid) {
        let group = system.switch_group(group_idx);
        let lg_m = hash::lg_modulus(group.modulus);
        if bulk_cache[group_idx].is_none() {
            bulk_cache[group_idx] = Some(hash::hash_bulk(
                ctrl_mask,
                group_idx,
                group.members.len() * lg_m,
            ));
        }
        let wide = bulk_cache[group_idx].as_ref().unwrap();
        hash::extract_ncf(wide, member_idx, group.modulus)
    } else {
        let Gate::Switch { out, .. } = system.gates[gid] else {
            unreachable!("switch_hash on non-switch gate {gid}");
        };
        hash::hash_solo(
            ctrl_mask,
            nonce_base + gid as u64,
            system.is_cf(out),
            system.modulus(out),
        )
    }
}

/// Seed phase shared by every entry point: constants and declared inputs.
/// Returns the mask table with exactly the seeded slots set.
fn seed_masks(
    system: &System,
    input_wires: &[Wire],
    input_masks: &[Label],
    delta: u128,
) -> Vec<Option<Label>> {
    assert_eq!(input_wires.len(), input_masks.len());

    let mut masks: Vec<Option<Label>> = vec![None; system.num_wires()];

    // Seed 1: constants. Mask = -c·Δ_R (CF) or -c (NCF).
    // Δ_R depends only on the modulus; cache it per k (constants are numerous —
    // every fold bit allocates zeros — and rebuilding Δ_R walks all λ coords).
    let mut delta_r_cache: [Option<Label>; 33] = [const { None }; 33];
    for (wid, slot) in masks.iter_mut().enumerate() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        let modulus = v.modulus;
        let neg_c = if v.v == 0 { 0 } else { modulus - v.v };
        *slot = Some(if system.is_cf_flags[wid] {
            let k = modulus.trailing_zeros() as usize;
            let d =
                delta_r_cache[k].get_or_insert_with(|| Label::Cf(label::delta_r(delta, modulus)));
            label::scalar_mul(neg_c, d)
        } else {
            Label::Ncf(NcfLabel {
                rep: neg_c,
                modulus,
            })
        });
    }

    // Seed 2: declared inputs (CF, caller-supplied).
    for (&w, m) in input_wires.iter().zip(input_masks.iter()) {
        assert!(system.is_cf(w), "input wire {} must be CF", w.wid);
        assert_eq!(
            m.modulus(),
            system.modulus(w),
            "input mask modulus mismatch"
        );
        assert!(m.is_cf(), "input mask must be CF");
        assert!(
            masks[w.wid].is_none(),
            "input wire {} already masked",
            w.wid
        );
        masks[w.wid] = Some(m.clone());
    }

    masks
}

/// Emission phase shared by every entry point: join diffs (the only
/// switch-system communication) and output masks in deterministic order.
/// Switches emit nothing — the evaluator derives every control from cleartext
/// `x` (paper §3.3).
fn emit_program(system: &System, masks: &[Option<Label>], output_wires: &[Wire]) -> Program {
    let mut program = Program::with_num_gates(system.num_gates());
    for (gid, &g) in system.gates.iter().enumerate() {
        if let Gate::Join { a: aw, b: bw } = g {
            let a = masks[aw.wid]
                .as_ref()
                .unwrap_or_else(|| panic!("join gate {}: a-side mask unresolved", gid));
            let b = masks[bw.wid]
                .as_ref()
                .unwrap_or_else(|| panic!("join gate {}: b-side mask unresolved", gid));
            program.set_join_diff(gid, label::sub(a, b));
        }
    }

    // Output masks: the streaming pipeline carries CF chunk-word masks across
    // phase boundaries; emission is kind-neutral.
    for &w in output_wires {
        let m = masks[w.wid]
            .clone()
            .unwrap_or_else(|| panic!("output wire {}: mask unresolved", w.wid));
        program.push_output_mask(m);
    }

    program
}

/// Garble `system`, exposing `input_wires` as evaluator-chosen inputs and
/// `output_wires` as NCF outputs she will recover.
pub fn garble(
    system: &System,
    input_wires: &[Wire],
    input_masks: &[Label],
    output_wires: &[Wire],
    delta: u128,
    nonce_base: u64,
) -> Program {
    garble_impl(
        system,
        input_wires,
        input_masks,
        output_wires,
        delta,
        None,
        nonce_base,
    )
}

/// [`garble`], additionally recording the mask-derivation tape.
///
/// The tape holds one [`JournalEntry`] per wire the propagation derived (gate
/// `gid` fired, setting `wid`), in derivation order; seeds (constants and
/// declared inputs) are not taped. The returned [`Program`] is identical to
/// [`garble`]'s.
///
/// The firing schedule depends only on the system's *structure* — gate kinds,
/// wire ids/subscriptions, and the Mul scalar zero/nonzero pattern — never on
/// mask values, so the `arena` module compiles the tape once per shape and runs
/// it for every structurally identical phase.
pub fn garble_recorded(
    system: &System,
    input_wires: &[Wire],
    input_masks: &[Label],
    output_wires: &[Wire],
    delta: u128,
    nonce_base: u64,
) -> (Program, Vec<JournalEntry>) {
    assert!(
        system.num_wires() <= u32::MAX as usize && system.num_gates() <= u32::MAX as usize,
        "journal entries use u32 ids"
    );
    let mut tape = Vec::new();
    let program = garble_impl(
        system,
        input_wires,
        input_masks,
        output_wires,
        delta,
        Some(&mut tape),
        nonce_base,
    );
    (program, tape)
}

/// Shared engine behind [`garble`] / [`garble_recorded`]: worklist propagation
/// to fixpoint, optionally recording each derived wire onto `tape`.
fn garble_impl(
    system: &System,
    input_wires: &[Wire],
    input_masks: &[Label],
    output_wires: &[Wire],
    delta: u128,
    mut tape: Option<&mut Vec<JournalEntry>>,
    nonce_base: u64,
) -> Program {
    let mut masks = seed_masks(system, input_wires, input_masks, delta);
    let mut wl = Worklist::new(system.num_wires(), system.num_gates());

    // Joins never fire during propagation — the diff is emitted in the final
    // pass — so they are done before the worklist starts.
    for (gid, g) in system.gates.iter().enumerate() {
        if matches!(g, Gate::Join { .. }) {
            wl.mark_done(gid);
        }
    }

    // Worklist seed: subscribers of every seeded wire, plus all `Mul(0, _)`
    // gates — the one gate shape that fires with no masked input (its output
    // mask is `0` unconditionally; that's what lets the phantom `bs[i]` wires
    // in `word_to_hot_with_bits` get their first mask). Every other gate needs
    // at least one known wire, so it is reachable through `try_set` cascades.
    let mut bulk_cache: BulkCache = vec![None; system.num_switch_groups()];
    let subs = system.subscriptions();
    for (wid, m) in masks.iter().enumerate() {
        if m.is_some() {
            wl.mark_known(wid);
            wl.enqueue_all(subs.of(wid));
        }
    }
    for (gid, g) in system.gates.iter().enumerate() {
        if matches!(g, Gate::Mul { scalar: 0, .. }) {
            wl.enqueue(gid);
        }
    }
    while let Some(gid) = wl.pop_live() {
        if propagate_gate(
            system,
            gid,
            &mut masks,
            &mut wl,
            &mut bulk_cache,
            tape.as_deref_mut(),
            nonce_base,
        ) {
            wl.mark_done(gid);
        }
    }

    emit_program(system, &masks, output_wires)
}

/// Fire gate `gid` once; returns true when the gate can never derive a new
/// wire. Invariant: `wl.wire_known(w)` ⇔ `masks[w].is_some()` — definedness is
/// read from the bitset; `masks` is loaded only for operands.
#[allow(clippy::too_many_arguments)]
fn propagate_gate(
    system: &System,
    gid: GateId,
    masks: &mut [Option<Label>],
    wl: &mut Worklist,
    bulk_cache: &mut BulkCache,
    mut tape: Option<&mut Vec<JournalEntry>>,
    nonce_base: u64,
) -> bool {
    match system.gates[gid] {
        Gate::Add { in0, in1, out } => {
            // out = in0 + in1  ⇔  in0 = out - in1  ⇔  in1 = out - in0
            // Compute a direction only if its target is still unset: label ops
            // allocate and (for k>1) walk all λ coordinates, so speculative
            // recomputation on every wakeup dominated garble time.
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                let v = label::add(
                    masks[in0.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::sub(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, in0, v, gid, wl, system, tape.as_deref_mut());
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                let v = label::sub(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in0.wid].as_ref().unwrap(),
                );
                try_set(masks, in1, v, gid, wl, system, tape.as_deref_mut());
            }
            wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
        }
        Gate::Sub { in0, in1, out } => {
            // out = in0 - in1  ⇔  in0 = out + in1  ⇔  in1 = in0 - out
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                let v = label::sub(
                    masks[in0.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::add(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, in0, v, gid, wl, system, tape.as_deref_mut());
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                let v = label::sub(
                    masks[in0.wid].as_ref().unwrap(),
                    masks[out.wid].as_ref().unwrap(),
                );
                try_set(masks, in1, v, gid, wl, system, tape.as_deref_mut());
            }
            wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
        }
        Gate::Mul { in0, scalar, out } => {
            // X_out = s · X_in. When s = 0 this is 0 regardless of X_in, so the
            // gate is fireable without its input being masked.
            if !wl.wire_known(out.wid) {
                if scalar == 0 {
                    try_set(
                        masks,
                        out,
                        Label::zero(system.is_cf(out), system.modulus(out)),
                        gid,
                        wl,
                        system,
                        tape.as_deref_mut(),
                    );
                } else if wl.wire_known(in0.wid) {
                    let v = label::scalar_mul(scalar, masks[in0.wid].as_ref().unwrap());
                    try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
                }
            }
            wl.wire_known(out.wid)
        }
        Gate::Mod2k { in0, k, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::mod2k(masks[in0.wid].as_ref().unwrap(), k);
                try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
            }
            wl.wire_known(out.wid)
        }
        Gate::Div2k { in0, k, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::div2k(masks[in0.wid].as_ref().unwrap(), k);
                try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
            }
            wl.wire_known(out.wid)
        }
        Gate::Switch { data, ctrl, out } => {
            // out = H(ctrl, gid) + data  ⇔  data = out - H(ctrl, gid).
            // For grouped switches, H is sliced from a single wide bulk call.
            // Skip the hash entirely once both sides are determined.
            let need_out = !wl.wire_known(out.wid) && wl.wire_known(data.wid);
            let need_data = !wl.wire_known(data.wid) && wl.wire_known(out.wid);
            if (need_out || need_data) && wl.wire_known(ctrl.wid) {
                let h = switch_hash(
                    system,
                    gid,
                    masks[ctrl.wid].as_ref().unwrap(),
                    bulk_cache,
                    nonce_base,
                );
                if need_out {
                    let v = label::add(&h, masks[data.wid].as_ref().unwrap());
                    try_set(masks, out, v, gid, wl, system, tape.as_deref_mut());
                }
                if need_data {
                    let v = label::sub(masks[out.wid].as_ref().unwrap(), &h);
                    try_set(masks, data, v, gid, wl, system, tape.as_deref_mut());
                }
            }
            // Garbler never branches on the control value: done once both
            // sides are determined.
            wl.wire_known(data.wid) && wl.wire_known(out.wid)
        }
        Gate::Join { .. } => {
            // No mask propagation; the join diff is emitted in the final pass.
            // Marked done at init, so this arm is never reached from the queue.
            true
        }
        Gate::SameWire { a, b } => {
            if !wl.wire_known(b.wid) && wl.wire_known(a.wid) {
                let v = masks[a.wid].clone().unwrap();
                try_set(masks, b, v, gid, wl, system, tape.as_deref_mut());
            }
            if !wl.wire_known(a.wid) && wl.wire_known(b.wid) {
                let v = masks[b.wid].clone().unwrap();
                try_set(masks, a, v, gid, wl, system, tape);
            }
            wl.wire_known(a.wid) && wl.wire_known(b.wid)
        }
    }
}

/// Set `w`'s mask if still unset, wake its subscribers, and (when recording)
/// append the derivation to `tape` — only writes that actually happen are
/// taped, so the tape is exactly the firing schedule the compiled garbler
/// (the `arena` module) needs.
fn try_set(
    masks: &mut [Option<Label>],
    w: Wire,
    new_mask: Label,
    src: GateId,
    wl: &mut Worklist,
    system: &System,
    tape: Option<&mut Vec<JournalEntry>>,
) {
    if !wl.wire_known(w.wid) {
        assert_eq!(
            new_mask.modulus(),
            system.modulus(w),
            "modulus mismatch setting wire {}",
            w.wid
        );
        masks[w.wid] = Some(new_mask);
        if let Some(t) = tape {
            t.push(JournalEntry {
                gid: src as u32,
                wid: w.wid as u32,
            });
        }
        wl.mark_known(w.wid);
        wl.wake_subscribers(system.subscriptions().of(w.wid), src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp_gc::convert::{
        bin_to_word, compute_sub_widths, fold_to_mod_ohe, sub_chunk_extract,
    };
    use crate::pipeline::sample_cf_mask;
    use rand::Rng;

    fn rng() -> impl Rng {
        rand::rng()
    }

    /// `Program` has no `PartialEq`; compare its three observable surfaces:
    /// per-gate join diffs, output masks (in order), and total bit count.
    fn assert_programs_equal(system: &System, a: &Program, b: &Program, ctx: &str) {
        assert_eq!(a.total_bits(), b.total_bits(), "{ctx}: total_bits");
        assert_eq!(a.output_masks(), b.output_masks(), "{ctx}: output masks");
        for gid in 0..system.num_gates() {
            assert_eq!(
                a.join_diff(gid),
                b.join_diff(gid),
                "{ctx}: join diff at gate {gid}"
            );
        }
    }

    /// The real header shape: `bin_to_word` → `sub_chunk_extract` (circular
    /// `word_to_hot`, `Mul(0)` seeding, backward propagation) →
    /// `fold_to_mod_ohe` (switch + join heavy).
    fn build_bits_header(ell: u32, p: u64) -> (System, Vec<Wire>, Vec<Wire>) {
        let sub_widths = compute_sub_widths(ell, 4); // two sub-chunks: forces a peel
        let mut sys = System::new();
        let bit_wires: Vec<Wire> = (0..ell).map(|_| sys.input(2)).collect();
        let x = bin_to_word(&mut sys, &bit_wires, ell);
        let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);
        let h_p = fold_to_mod_ohe(&mut sys, &extraction, p);
        (sys, bit_wires, h_p)
    }

    #[test]
    fn test_garble_recorded_matches_garble() {
        // garble_recorded must emit the identical Program (and replaying its
        // own tape must too), across primes, masks and deltas.
        let mut rng = rng();
        let ell: u32 = 8;
        for p in [3u64, 7] {
            for round in 0..3 {
                let (sys, bit_wires, h_p) = build_bits_header(ell, p);
                let input_masks: Vec<Label> = bit_wires
                    .iter()
                    .map(|_| sample_cf_mask(&mut rng, 2))
                    .collect();
                let delta: u128 = rng.random();

                let expected = garble(&sys, &bit_wires, &input_masks, &h_p, delta, 0);
                let (recorded, tape) =
                    garble_recorded(&sys, &bit_wires, &input_masks, &h_p, delta, 0);
                assert_programs_equal(&sys, &expected, &recorded, &format!("p={p} r={round}"));
                assert!(!tape.is_empty(), "header derivation must tape entries");
            }
        }
    }

}
