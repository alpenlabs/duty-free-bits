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

use super::program::Program;
use crate::exec::Worklist;
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
        hash::hash_solo(ctrl_mask, gid, system.is_cf(out), system.modulus(out))
    }
}

/// Garble `system`, exposing `input_wires` as evaluator-chosen inputs and
/// `output_wires` as NCF outputs she will recover.
pub fn garble(
    system: &System,
    input_wires: &[Wire],
    input_masks: &[Label],
    output_wires: &[Wire],
    delta: u128,
) -> Program {
    assert_eq!(input_wires.len(), input_masks.len());

    let mut masks: Vec<Option<Label>> = vec![None; system.num_wires()];
    let mut wl = Worklist::new(system.num_wires(), system.num_gates());

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
        wl.mark_known(wid);
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
        wl.mark_known(w.wid);
    }

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
    for wid in 0..system.num_wires() {
        if wl.wire_known(wid) {
            wl.enqueue_all(subs.of(wid));
        }
    }
    for (gid, g) in system.gates.iter().enumerate() {
        if matches!(g, Gate::Mul { scalar: 0, .. }) {
            wl.enqueue(gid);
        }
    }
    while let Some(gid) = wl.pop_live() {
        if propagate_gate(system, gid, &mut masks, &mut wl, &mut bulk_cache) {
            wl.mark_done(gid);
        }
    }

    // Second pass: emit join diffs (the only switch-system communication) and
    // output masks in deterministic order. Switches emit nothing — the evaluator
    // derives every control from cleartext `x` (paper §3.3).
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

/// Fire gate `gid` once; returns true when the gate can never derive a new
/// wire. Invariant: `wl.wire_known(w)` ⇔ `masks[w].is_some()` — definedness is
/// read from the bitset; `masks` is loaded only for operands.
fn propagate_gate(
    system: &System,
    gid: GateId,
    masks: &mut [Option<Label>],
    wl: &mut Worklist,
    bulk_cache: &mut BulkCache,
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
                try_set(masks, out, v, gid, wl, system);
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::sub(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, in0, v, gid, wl, system);
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                let v = label::sub(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in0.wid].as_ref().unwrap(),
                );
                try_set(masks, in1, v, gid, wl, system);
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
                try_set(masks, out, v, gid, wl, system);
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::add(
                    masks[out.wid].as_ref().unwrap(),
                    masks[in1.wid].as_ref().unwrap(),
                );
                try_set(masks, in0, v, gid, wl, system);
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                let v = label::sub(
                    masks[in0.wid].as_ref().unwrap(),
                    masks[out.wid].as_ref().unwrap(),
                );
                try_set(masks, in1, v, gid, wl, system);
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
                    );
                } else if wl.wire_known(in0.wid) {
                    let v = label::scalar_mul(scalar, masks[in0.wid].as_ref().unwrap());
                    try_set(masks, out, v, gid, wl, system);
                }
            }
            wl.wire_known(out.wid)
        }
        Gate::Mod2k { in0, k, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::mod2k(masks[in0.wid].as_ref().unwrap(), k);
                try_set(masks, out, v, gid, wl, system);
            }
            wl.wire_known(out.wid)
        }
        Gate::Div2k { in0, k, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::div2k(masks[in0.wid].as_ref().unwrap(), k);
                try_set(masks, out, v, gid, wl, system);
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
                let h = switch_hash(system, gid, masks[ctrl.wid].as_ref().unwrap(), bulk_cache);
                if need_out {
                    let v = label::add(&h, masks[data.wid].as_ref().unwrap());
                    try_set(masks, out, v, gid, wl, system);
                }
                if need_data {
                    let v = label::sub(masks[out.wid].as_ref().unwrap(), &h);
                    try_set(masks, data, v, gid, wl, system);
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
                try_set(masks, b, v, gid, wl, system);
            }
            if !wl.wire_known(a.wid) && wl.wire_known(b.wid) {
                let v = masks[b.wid].clone().unwrap();
                try_set(masks, a, v, gid, wl, system);
            }
            wl.wire_known(a.wid) && wl.wire_known(b.wid)
        }
    }
}

fn try_set(
    masks: &mut [Option<Label>],
    w: Wire,
    new_mask: Label,
    src: GateId,
    wl: &mut Worklist,
    system: &System,
) {
    if !wl.wire_known(w.wid) {
        assert_eq!(
            new_mask.modulus(),
            system.modulus(w),
            "modulus mismatch setting wire {}",
            w.wid
        );
        masks[w.wid] = Some(new_mask);
        wl.mark_known(w.wid);
        wl.wake_subscribers(system.subscriptions().of(w.wid), src);
    }
}
