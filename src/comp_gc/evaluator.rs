//! Evaluator: decode labels using a worklist, mirroring `Exec`'s structure.
//!
//! Each wire's evaluator label ultimately takes the form `X + x·Δ_R` (CF) or
//! `X + x` (NCF), where `X` is the garbler's mask. Labels are derived in *any*
//! order the graph permits — forward through affine gates, backward too, and
//! through switches only when `ctrl = 0`. The control is taken from the
//! cleartext wire `values`.

use super::program::Program;
use crate::exec::{JournalEntry, Worklist};
use crate::hash;
use crate::label::{self, Label};
use crate::system::System;
use crate::types::{Gate, GateId, Val, Wire};

/// Lazy cache for per-group bulk-hash outputs (mirrors the garbler's cache).
type BulkCache = Vec<Option<Vec<u8>>>;

/// Resolve the H value for a switch gate, packing across group members
/// when the gate is registered in a [`SwitchGroup`].
fn switch_hash(
    system: &System,
    gid: GateId,
    ctrl_label: &Label,
    bulk_cache: &mut BulkCache,
) -> Label {
    if let Some((group_idx, member_idx)) = system.gate_group(gid) {
        let group = system.switch_group(group_idx);
        let lg_m = hash::lg_modulus(group.modulus);
        if bulk_cache[group_idx].is_none() {
            bulk_cache[group_idx] = Some(hash::hash_bulk(
                ctrl_label,
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
        hash::hash_solo(ctrl_label, gid, system.is_cf(out), system.modulus(out))
    }
}

/// Evaluate a garbled system at the label level.
///
/// Takes the evaluator's labels for inputs (already including value·Δ_R) plus the
/// cleartext `values` of every wire (from running [`Exec`] on the evaluator's
/// known input `x`), and returns the labels of the requested output wires. The
/// cleartext values supply switch controls.
pub fn eval_with_labels(
    system: &System,
    input_wires: &[Wire],
    input_labels: &[Label],
    values: &[Val],
    program: &Program,
    output_wires: &[Wire],
) -> Vec<Label> {
    assert_eq!(input_wires.len(), input_labels.len());
    assert_eq!(
        values.len(),
        system.num_wires(),
        "values must cover all wires"
    );

    let mut labels: Vec<Option<Label>> = vec![None; system.num_wires()];
    let mut wl = Worklist::new(system.num_wires(), system.num_gates());

    // Constants: evaluator's label is 0 in the appropriate ring.
    for (wid, slot) in labels.iter_mut().enumerate() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        *slot = Some(Label::zero(system.is_cf_flags[wid], v.modulus));
        wl.mark_known(wid);
    }

    // Inputs: caller-provided labels.
    for (&w, lbl) in input_wires.iter().zip(input_labels.iter()) {
        assert_eq!(
            lbl.modulus(),
            system.modulus(w),
            "input label modulus mismatch"
        );
        labels[w.wid] = Some(lbl.clone());
        wl.mark_known(w.wid);
    }

    // Seed worklist: every gate subscribed to a currently-labeled wire.
    let subs = system.subscriptions();
    for wid in 0..system.num_wires() {
        if wl.wire_known(wid) {
            wl.enqueue_all(subs.of(wid));
        }
    }

    // Propagate.
    let mut bulk_cache: BulkCache = vec![None; system.num_switch_groups()];
    while let Some(gid) = wl.pop_live() {
        if fire_gate(
            system,
            gid,
            &mut labels,
            values,
            program,
            &mut wl,
            &mut bulk_cache,
        ) {
            wl.mark_done(gid);
        }
    }

    // Pull out labels for the requested outputs.
    output_wires
        .iter()
        .map(|&w| {
            labels[w.wid]
                .clone()
                .unwrap_or_else(|| panic!("no label on output wire {}", w.wid))
        })
        .collect()
}

/// Evaluate a garbled system by replaying a cleartext-execution journal.
///
/// `journal` is the set-order record of an [`Exec::run_recorded`] pass
/// (`crate::exec`): step n says gate `gid` derived wire `wid`. Label
/// propagation derives wires through exactly the same gate directions as value
/// propagation — joins via the program diff, switches only when the cleartext
/// control is 0 — so by induction every operand a step reads is label-known
/// when the step executes, and the result equals [`eval_with_labels`]'s
/// fixpoint on every value-derivable wire (which includes every output and
/// every operand). This replaces the worklist with a linear tape: no queue, no
/// wakeups, no definedness checks.
///
/// [`Exec::run_recorded`]: crate::exec::Exec::run_recorded
pub fn replay_with_labels(
    system: &System,
    input_wires: &[Wire],
    input_labels: &[Label],
    values: &[Val],
    program: &Program,
    journal: &[JournalEntry],
    output_wires: &[Wire],
) -> Vec<Label> {
    assert_eq!(input_wires.len(), input_labels.len());
    assert_eq!(
        values.len(),
        system.num_wires(),
        "values must cover all wires"
    );

    let mut labels: Vec<Option<Label>> = vec![None; system.num_wires()];

    // Constants: evaluator's label is 0 in the appropriate ring.
    for (wid, slot) in labels.iter_mut().enumerate() {
        let v = system.values[wid];
        if v.defined {
            *slot = Some(Label::zero(system.is_cf_flags[wid], v.modulus));
        }
    }
    // Inputs: caller-provided labels.
    for (&w, lbl) in input_wires.iter().zip(input_labels.iter()) {
        assert_eq!(
            lbl.modulus(),
            system.modulus(w),
            "input label modulus mismatch"
        );
        labels[w.wid] = Some(lbl.clone());
    }

    let mut bulk_cache: BulkCache = vec![None; system.num_switch_groups()];
    for &JournalEntry { gid, wid } in journal {
        let gid = gid as usize;
        let wid = wid as usize;
        // Operands are borrowed (the derived label is written only after the
        // arm computes it); cloning here heap-copied every lane operand.
        let lbl = |w: Wire| -> &Label {
            labels[w.wid]
                .as_ref()
                .unwrap_or_else(|| panic!("replay: operand wire {} unlabeled", w.wid))
        };
        let v = match system.gates[gid] {
            Gate::Add { in0, in1, out } => {
                if wid == out.wid {
                    label::add(lbl(in0), lbl(in1)) // out = in0 + in1
                } else if wid == in0.wid {
                    label::sub(lbl(out), lbl(in1)) // in0 = out - in1
                } else {
                    label::sub(lbl(out), lbl(in0)) // in1 = out - in0
                }
            }
            Gate::Sub { in0, in1, out } => {
                if wid == out.wid {
                    label::sub(lbl(in0), lbl(in1)) // out = in0 - in1
                } else if wid == in0.wid {
                    label::add(lbl(out), lbl(in1)) // in0 = out + in1
                } else {
                    label::sub(lbl(in0), lbl(out)) // in1 = in0 - out
                }
            }
            Gate::Mul { in0, scalar, out } => {
                debug_assert_eq!(wid, out.wid);
                if scalar == 0 {
                    Label::zero(system.is_cf(out), system.modulus(out))
                } else {
                    label::scalar_mul(scalar, lbl(in0))
                }
            }
            Gate::Mod2k { in0, k, out } => {
                debug_assert_eq!(wid, out.wid);
                label::mod2k(lbl(in0), k)
            }
            Gate::Div2k { in0, k, out } => {
                debug_assert_eq!(wid, out.wid);
                label::div2k(lbl(in0), k)
            }
            Gate::Switch { data, ctrl, out } => {
                // Exec fires a switch only when its control value is 0; a
                // journal from a different Exec/input would silently produce
                // garbage labels here, so check unconditionally.
                assert_eq!(values[ctrl.wid].v, 0, "replay: switch {gid} open");
                let h = switch_hash(system, gid, lbl(ctrl), &mut bulk_cache);
                if wid == out.wid {
                    label::add(lbl(data), &h) // out = data + H
                } else {
                    label::sub(lbl(out), &h) // data = out - H
                }
            }
            Gate::Join { a: aw, b: bw } => {
                // diff = X_a - X_b ⇒ label_b = label_a - diff, label_a = label_b + diff.
                let diff = program
                    .join_diff(gid)
                    .unwrap_or_else(|| panic!("missing join diff for gate {}", gid));
                if wid == bw.wid {
                    label::sub(lbl(aw), diff)
                } else {
                    label::add(lbl(bw), diff)
                }
            }
            Gate::SameWire { a: aw, b: bw } => {
                if wid == bw.wid {
                    lbl(aw).clone()
                } else {
                    lbl(bw).clone()
                }
            }
        };
        debug_assert_eq!(
            v.modulus(),
            system.modulus(Wire { wid }),
            "modulus mismatch setting wire {wid}"
        );
        assert!(labels[wid].is_none(), "replay: wire {wid} already labeled");
        labels[wid] = Some(v);
    }

    output_wires
        .iter()
        .map(|&w| {
            labels[w.wid]
                .clone()
                .unwrap_or_else(|| panic!("no label on output wire {}", w.wid))
        })
        .collect()
}

/// Fire gate `gid` once; returns true when the gate can never derive a new
/// wire. Invariant: `wl.wire_known(w)` ⇔ `labels[w].is_some()` — definedness is
/// read from the bitset; `labels` is loaded only for operands.
fn fire_gate(
    system: &System,
    gid: GateId,
    labels: &mut [Option<Label>],
    values: &[Val],
    program: &Program,
    wl: &mut Worklist,
    bulk_cache: &mut BulkCache,
) -> bool {
    match system.gates[gid] {
        Gate::Add { in0, in1, out } => {
            // Compute a direction only if its target is still unset: label ops
            // allocate and (for k>1) walk all λ coordinates, so speculative
            // recomputation on every wakeup dominated eval time.
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                let v = label::add(
                    labels[in0.wid].as_ref().unwrap(),
                    labels[in1.wid].as_ref().unwrap(),
                ); // out = in0 + in1
                try_set(labels, out, v, gid, wl, system);
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::sub(
                    labels[out.wid].as_ref().unwrap(),
                    labels[in1.wid].as_ref().unwrap(),
                ); // in0 = out - in1
                try_set(labels, in0, v, gid, wl, system);
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::sub(
                    labels[out.wid].as_ref().unwrap(),
                    labels[in0.wid].as_ref().unwrap(),
                ); // in1 = out - in0
                try_set(labels, in1, v, gid, wl, system);
            }
            wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
        }
        Gate::Sub { in0, in1, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                let v = label::sub(
                    labels[in0.wid].as_ref().unwrap(),
                    labels[in1.wid].as_ref().unwrap(),
                ); // out = in0 - in1
                try_set(labels, out, v, gid, wl, system);
            }
            if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                let v = label::add(
                    labels[out.wid].as_ref().unwrap(),
                    labels[in1.wid].as_ref().unwrap(),
                ); // in0 = out + in1
                try_set(labels, in0, v, gid, wl, system);
            }
            if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                let v = label::sub(
                    labels[in0.wid].as_ref().unwrap(),
                    labels[out.wid].as_ref().unwrap(),
                ); // in1 = in0 - out
                try_set(labels, in1, v, gid, wl, system);
            }
            wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
        }
        Gate::Mul { in0, scalar, out } => {
            // label_out = s · label_in; when s = 0 the output label is 0 regardless.
            if !wl.wire_known(out.wid) {
                if scalar == 0 {
                    try_set(
                        labels,
                        out,
                        Label::zero(system.is_cf(out), system.modulus(out)),
                        gid,
                        wl,
                        system,
                    );
                } else if wl.wire_known(in0.wid) {
                    let v = label::scalar_mul(scalar, labels[in0.wid].as_ref().unwrap());
                    try_set(labels, out, v, gid, wl, system);
                }
            }
            wl.wire_known(out.wid)
        }
        Gate::Mod2k { in0, k, out } => {
            // Forward only; low-bit dropping is not invertible.
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::mod2k(labels[in0.wid].as_ref().unwrap(), k);
                try_set(labels, out, v, gid, wl, system);
            }
            wl.wire_known(out.wid)
        }
        Gate::Div2k { in0, k, out } => {
            if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                let v = label::div2k(labels[in0.wid].as_ref().unwrap(), k);
                try_set(labels, out, v, gid, wl, system);
            }
            wl.wire_known(out.wid)
        }
        Gate::Switch { data, ctrl, out } => {
            // The control value is known in cleartext (the evaluator knows `x`):
            // the switch fires iff ctrl = 0. Nothing is revealed by the garbler.
            let cv = values[ctrl.wid];
            debug_assert!(!cv.is_none(), "switch {gid}: ctrl value undefined");
            if cv.v != 0 {
                // `values` is fixed for the whole run: a defined non-zero
                // control means the switch is open forever.
                return !cv.is_none();
            }
            // Skip the hash entirely once both sides are determined.
            let need_out = !wl.wire_known(out.wid) && wl.wire_known(data.wid);
            let need_data = !wl.wire_known(data.wid) && wl.wire_known(out.wid);
            // ctrl = 0: we still need the ctrl *label* to form H. Wait for it.
            if (need_out || need_data) && wl.wire_known(ctrl.wid) {
                // For grouped switches, H is sliced from a single wide bulk call.
                let h = switch_hash(system, gid, labels[ctrl.wid].as_ref().unwrap(), bulk_cache);
                if need_out {
                    let v = label::add(labels[data.wid].as_ref().unwrap(), &h); // out = data + H
                    try_set(labels, out, v, gid, wl, system);
                }
                if need_data {
                    let v = label::sub(labels[out.wid].as_ref().unwrap(), &h); // data = out - H
                    try_set(labels, data, v, gid, wl, system);
                }
            }
            wl.wire_known(data.wid) && wl.wire_known(out.wid)
        }
        Gate::Join { a: aw, b: bw } => {
            // diff = X_a - X_b, so label_b = label_a - diff and label_a = label_b + diff.
            let ka = wl.wire_known(aw.wid);
            let kb = wl.wire_known(bw.wid);
            if ka != kb {
                let diff = program
                    .join_diff(gid)
                    .unwrap_or_else(|| panic!("missing join diff for gate {}", gid));
                if ka {
                    let v = label::sub(labels[aw.wid].as_ref().unwrap(), diff);
                    try_set(labels, bw, v, gid, wl, system);
                } else {
                    let v = label::add(labels[bw.wid].as_ref().unwrap(), diff);
                    try_set(labels, aw, v, gid, wl, system);
                }
            }
            wl.wire_known(aw.wid) && wl.wire_known(bw.wid)
        }
        Gate::SameWire { a: aw, b: bw } => {
            if !wl.wire_known(bw.wid) && wl.wire_known(aw.wid) {
                let v = labels[aw.wid].clone().unwrap();
                try_set(labels, bw, v, gid, wl, system);
            }
            if !wl.wire_known(aw.wid) && wl.wire_known(bw.wid) {
                let v = labels[bw.wid].clone().unwrap();
                try_set(labels, aw, v, gid, wl, system);
            }
            wl.wire_known(aw.wid) && wl.wire_known(bw.wid)
        }
    }
}

fn try_set(
    labels: &mut [Option<Label>],
    w: Wire,
    new_label: Label,
    src: GateId,
    wl: &mut Worklist,
    system: &System,
) {
    if !wl.wire_known(w.wid) {
        assert_eq!(
            new_label.modulus(),
            system.modulus(w),
            "modulus mismatch setting wire {}",
            w.wid
        );
        labels[w.wid] = Some(new_label);
        wl.mark_known(w.wid);
        wl.wake_subscribers(system.subscriptions().of(w.wid), src);
    }
}
