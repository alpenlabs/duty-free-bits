//! Evaluator: decode labels using a worklist, mirroring `Exec`'s structure.
//!
//! Each wire's evaluator label ultimately takes the form `X + x·Δ_R` (CF) or
//! `X + x` (NCF), where `X` is the garbler's mask. Labels are derived in *any*
//! order the graph permits — forward through affine gates, backward too, and
//! through switches only when `ctrl = 0`. The control is taken from the
//! cleartext wire `values` (the evaluator knows `x`, so it computes them itself —
//! paper §3.3: switches reveal nothing). This is what allows patterns like
//! `ohe_scale`'s "junk sh_i labels, join fixes the sum, backward add recovers
//! individual sh_i" to work.

use crate::hash;
use crate::label::{self, Label};
use super::program::Program;
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
/// cleartext values supply switch controls — nothing is revealed by the garbler.
/// No mask-subtraction or value decoding here; the caller converts labels back to
/// values when needed. This is the form used by [`Pipeline`] for streaming
/// garble+eval.
///
/// [`Exec`]: crate::exec::Exec
/// [`Pipeline`]: crate::pipeline::Pipeline
pub fn eval_with_labels(
    system: &System,
    input_wires: &[Wire],
    input_labels: &[Label],
    values: &[Val],
    delta: u128,
    program: &Program,
    output_wires: &[Wire],
) -> Vec<Label> {
    assert_eq!(input_wires.len(), input_labels.len());
    assert_eq!(values.len(), system.num_wires(), "values must cover all wires");
    assert_eq!(delta & 1, 1, "Δ must have low bit 1");

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
        assert_eq!(lbl.modulus(), system.modulus(w), "input label modulus mismatch");
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
        fire_gate(system, gid, &mut labels, values, program, &mut queue, &mut bulk_cache);
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
            let (l0, l1, lo) = (labels[in0.wid].clone(), labels[in1.wid].clone(), labels[out.wid].clone());
            if let (Some(a), Some(b)) = (&l0, &l1) {
                try_set(labels, out, label::add(a, b), queue, system); // out = in0 + in1
            }
            if let (Some(o), Some(b)) = (&lo, &l1) {
                try_set(labels, in0, label::sub(o, b), queue, system); // in0 = out - in1
            }
            if let (Some(o), Some(a)) = (&lo, &l0) {
                try_set(labels, in1, label::sub(o, a), queue, system); // in1 = out - in0
            }
        }
        Gate::Sub { in0, in1, out } => {
            let (l0, l1, lo) = (labels[in0.wid].clone(), labels[in1.wid].clone(), labels[out.wid].clone());
            if let (Some(a), Some(b)) = (&l0, &l1) {
                try_set(labels, out, label::sub(a, b), queue, system); // out = in0 - in1
            }
            if let (Some(o), Some(b)) = (&lo, &l1) {
                try_set(labels, in0, label::add(o, b), queue, system); // in0 = out + in1
            }
            if let (Some(a), Some(o)) = (&l0, &lo) {
                try_set(labels, in1, label::sub(a, o), queue, system); // in1 = in0 - out
            }
        }
        Gate::Mul { in0, scalar, out } => {
            // label_out = s · label_in; when s = 0 the output label is 0 regardless.
            if scalar == 0 {
                try_set(labels, out, Label::zero(system.is_cf(out), system.modulus(out)), queue, system);
            } else if let Some(a) = labels[in0.wid].clone() {
                try_set(labels, out, label::scalar_mul(scalar, &a), queue, system);
            }
        }
        Gate::Mod2k { in0, k, out } => {
            // Forward only; low-bit dropping is not invertible.
            if let Some(a) = labels[in0.wid].clone() {
                try_set(labels, out, label::mod2k(&a, k), queue, system);
            }
        }
        Gate::Div2k { in0, k, out } => {
            if let Some(a) = labels[in0.wid].clone() {
                try_set(labels, out, label::div2k(&a, k), queue, system);
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
            // ctrl = 0: we still need the ctrl *label* to form H. Wait for it.
            let Some(ctrl_label) = labels[ctrl.wid].clone() else {
                return;
            };
            // For grouped switches, H is sliced from a single wide bulk call.
            let h = switch_hash(system, gid, &ctrl_label, bulk_cache);
            if let Some(d) = labels[data.wid].clone() {
                try_set(labels, out, label::add(&d, &h), queue, system); // out = data + H
            }
            if let Some(o) = labels[out.wid].clone() {
                try_set(labels, data, label::sub(&o, &h), queue, system); // data = out - H
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
