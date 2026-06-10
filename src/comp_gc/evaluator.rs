//! Evaluator: decode labels using a worklist, mirroring `Exec`'s structure.
//!
//! Each wire's evaluator label ultimately takes the form `X + x·Δ_R` (CF) or
//! `X + x` (NCF), where `X` is the garbler's mask. Labels are derived in *any*
//! order the graph permits — forward through affine gates, backward too, and
//! through switches only when `ctrl = 0`. The control is taken from the
//! cleartext wire `values`.

use super::program::Program;
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

    // Constants: evaluator's label is 0 in the appropriate ring.
    for (wid, slot) in labels.iter_mut().enumerate() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        *slot = Some(Label::zero(system.is_cf_flags[wid], v.modulus));
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

    // Seed worklist: every gate subscribed to a currently-labeled wire.
    let mut queue: Vec<GateId> = Vec::new();
    for (wid, slot) in labels.iter().enumerate() {
        if slot.is_some() {
            queue.extend_from_slice(&system.subscriptions[wid]);
        }
    }

    // Propagate.
    let mut bulk_cache: BulkCache = vec![None; system.num_switch_groups()];
    while let Some(gid) = queue.pop() {
        fire_gate(
            system,
            gid,
            &mut labels,
            values,
            program,
            &mut queue,
            &mut bulk_cache,
        );
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

fn fire_gate(
    system: &System,
    gid: GateId,
    labels: &mut [Option<Label>],
    values: &[Val],
    program: &Program,
    queue: &mut Vec<GateId>,
    bulk_cache: &mut BulkCache,
) {
    match system.gates[gid] {
        Gate::Add { in0, in1, out } => {
            // Compute a direction only if its target is still unset: label ops
            // allocate and (for k>1) walk all λ coordinates, so speculative
            // recomputation on every wakeup dominated eval time.
            if labels[out.wid].is_none()
                && let (Some(a), Some(b)) = (&labels[in0.wid], &labels[in1.wid])
            {
                let v = label::add(a, b); // out = in0 + in1
                try_set(labels, out, v, queue, system);
            }
            if labels[in0.wid].is_none()
                && let (Some(o), Some(b)) = (&labels[out.wid], &labels[in1.wid])
            {
                let v = label::sub(o, b); // in0 = out - in1
                try_set(labels, in0, v, queue, system);
            }
            if labels[in1.wid].is_none()
                && let (Some(o), Some(a)) = (&labels[out.wid], &labels[in0.wid])
            {
                let v = label::sub(o, a); // in1 = out - in0
                try_set(labels, in1, v, queue, system);
            }
        }
        Gate::Sub { in0, in1, out } => {
            if labels[out.wid].is_none()
                && let (Some(a), Some(b)) = (&labels[in0.wid], &labels[in1.wid])
            {
                let v = label::sub(a, b); // out = in0 - in1
                try_set(labels, out, v, queue, system);
            }
            if labels[in0.wid].is_none()
                && let (Some(o), Some(b)) = (&labels[out.wid], &labels[in1.wid])
            {
                let v = label::add(o, b); // in0 = out + in1
                try_set(labels, in0, v, queue, system);
            }
            if labels[in1.wid].is_none()
                && let (Some(a), Some(o)) = (&labels[in0.wid], &labels[out.wid])
            {
                let v = label::sub(a, o); // in1 = in0 - out
                try_set(labels, in1, v, queue, system);
            }
        }
        Gate::Mul { in0, scalar, out } => {
            // label_out = s · label_in; when s = 0 the output label is 0 regardless.
            if labels[out.wid].is_some() {
                return;
            }
            if scalar == 0 {
                try_set(
                    labels,
                    out,
                    Label::zero(system.is_cf(out), system.modulus(out)),
                    queue,
                    system,
                );
            } else if let Some(a) = &labels[in0.wid] {
                let v = label::scalar_mul(scalar, a);
                try_set(labels, out, v, queue, system);
            }
        }
        Gate::Mod2k { in0, k, out } => {
            // Forward only; low-bit dropping is not invertible.
            if labels[out.wid].is_none()
                && let Some(a) = &labels[in0.wid]
            {
                let v = label::mod2k(a, k);
                try_set(labels, out, v, queue, system);
            }
        }
        Gate::Div2k { in0, k, out } => {
            if labels[out.wid].is_none()
                && let Some(a) = &labels[in0.wid]
            {
                let v = label::div2k(a, k);
                try_set(labels, out, v, queue, system);
            }
        }
        Gate::Switch { data, ctrl, out } => {
            // The control value is known in cleartext (the evaluator knows `x`):
            // the switch fires iff ctrl = 0. Nothing is revealed by the garbler.
            let cv = values[ctrl.wid];
            debug_assert!(!cv.is_none(), "switch {gid}: ctrl value undefined");
            if cv.v != 0 {
                return; // switch open: no label propagation
            }
            // Skip the hash entirely once both sides are determined.
            let need_out = labels[out.wid].is_none() && labels[data.wid].is_some();
            let need_data = labels[data.wid].is_none() && labels[out.wid].is_some();
            if !(need_out || need_data) {
                return;
            }
            // ctrl = 0: we still need the ctrl *label* to form H. Wait for it.
            let Some(ctrl_label) = labels[ctrl.wid].clone() else {
                return;
            };
            // For grouped switches, H is sliced from a single wide bulk call.
            let h = switch_hash(system, gid, &ctrl_label, bulk_cache);
            if need_out && let Some(d) = &labels[data.wid] {
                let v = label::add(d, &h); // out = data + H
                try_set(labels, out, v, queue, system);
            }
            if need_data && let Some(o) = &labels[out.wid] {
                let v = label::sub(o, &h); // data = out - H
                try_set(labels, data, v, queue, system);
            }
        }
        Gate::Join { a: aw, b: bw } => {
            let diff = program
                .join_diff(gid)
                .unwrap_or_else(|| panic!("missing join diff for gate {}", gid))
                .clone();
            // diff = X_a - X_b, so label_b = label_a - diff and label_a = label_b + diff.
            let (la, lb) = (labels[aw.wid].clone(), labels[bw.wid].clone());
            if let Some(a) = &la {
                try_set(labels, bw, label::sub(a, &diff), queue, system);
            }
            if let Some(b) = &lb {
                try_set(labels, aw, label::add(b, &diff), queue, system);
            }
        }
        Gate::SameWire { a: aw, b: bw } => {
            let (la, lb) = (labels[aw.wid].clone(), labels[bw.wid].clone());
            if let Some(a) = la {
                try_set(labels, bw, a, queue, system);
            }
            if let Some(b) = lb {
                try_set(labels, aw, b, queue, system);
            }
        }
    }
}

fn try_set(
    labels: &mut [Option<Label>],
    w: Wire,
    new_label: Label,
    queue: &mut Vec<GateId>,
    system: &System,
) {
    if labels[w.wid].is_none() {
        assert_eq!(
            new_label.modulus(),
            system.modulus(w),
            "modulus mismatch setting wire {}",
            w.wid
        );
        labels[w.wid] = Some(new_label);
        queue.extend_from_slice(&system.subscriptions[w.wid]);
    }
}
