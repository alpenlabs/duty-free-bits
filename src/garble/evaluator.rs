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

use super::hash;
use super::label::{self, Label};
use super::program::Program;
use crate::system::System;
use crate::types::{GateId, GateType, Val, Wire};

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
    let g = system.gates[gid];
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
        hash::hash_solo(ctrl_label, gid, system.is_cf(g.out), system.modulus(g.out))
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
/// [`Pipeline`]: crate::garble::pipeline::Pipeline
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
    let g = system.gates[gid];
    let lin0 = labels[g.in0.wid].clone();
    let lin1 = labels[g.in1.wid].clone();
    // `out` is only a real wire for arithmetic / switch gates.
    let lout = match g.typ {
        GateType::Add
        | GateType::Sub
        | GateType::Mul
        | GateType::Mod2k
        | GateType::Div2k
        | GateType::Switch => labels[g.out.wid].clone(),
        _ => None,
    };

    match g.typ {
        GateType::Add => {
            // Forward: out = in0 + in1.
            if let (Some(a), Some(b)) = (&lin0, &lin1) {
                try_set(labels, g.out, label::add(a, b), queue, system);
            }
            // Backward: in0 = out - in1.
            if let (Some(o), Some(b)) = (&lout, &lin1) {
                try_set(labels, g.in0, label::sub(o, b), queue, system);
            }
            // Backward: in1 = out - in0.
            if let (Some(o), Some(a)) = (&lout, &lin0) {
                try_set(labels, g.in1, label::sub(o, a), queue, system);
            }
        }
        GateType::Sub => {
            // Forward: out = in0 - in1.
            if let (Some(a), Some(b)) = (&lin0, &lin1) {
                try_set(labels, g.out, label::sub(a, b), queue, system);
            }
            // Backward: in0 = out + in1.
            if let (Some(o), Some(b)) = (&lout, &lin1) {
                try_set(labels, g.in0, label::add(o, b), queue, system);
            }
            // Backward: in1 = in0 - out.
            if let (Some(a), Some(o)) = (&lin0, &lout) {
                try_set(labels, g.in1, label::sub(a, o), queue, system);
            }
        }
        GateType::Mul => {
            // label_out = s · label_in. Same zero-input-free rule as the
            // garbler: when s = 0, the output label is 0.
            if g.param == 0 {
                try_set(labels, g.out, Label::zero(system.is_cf(g.out), system.modulus(g.out)), queue, system);
            } else if let Some(a) = &lin0 {
                try_set(labels, g.out, label::scalar_mul(g.param, a), queue, system);
            }
        }
        GateType::Mod2k => {
            // Forward only; low-bit dropping is not invertible.
            if let Some(a) = &lin0 {
                try_set(labels, g.out, label::mod2k(a, g.param as u32), queue, system);
            }
        }
        GateType::Div2k => {
            // Forward only.
            if let Some(a) = &lin0 {
                try_set(labels, g.out, label::div2k(a, g.param as u32), queue, system);
            }
        }
        GateType::Switch => {
            // The control value is known in cleartext (the evaluator knows `x`):
            // the switch fires iff ctrl = 0. Nothing is revealed by the garbler.
            let ctrl = values[g.in1.wid];
            debug_assert!(!ctrl.is_none(), "switch {gid}: ctrl value undefined");
            if ctrl.v != 0 {
                // Switch does not fire: no label propagation.
                return;
            }
            // ctrl = 0: we still need the ctrl *label* to form H. Wait for it.
            let Some(ctrl_label) = lin1 else { return; };
            // Propagate in either direction via H. For grouped switches, H is
            // sliced from a single wide bulk call.
            let h = switch_hash(system, gid, &ctrl_label, bulk_cache);
            // Forward: out = in0 + H.
            if let Some(din) = &lin0 {
                try_set(labels, g.out, label::add(din, &h), queue, system);
            }
            // Backward: in0 = out - H.
            if let Some(o) = &lout {
                try_set(labels, g.in0, label::sub(o, &h), queue, system);
            }
        }
        GateType::Join => {
            let diff = program
                .join_diff(gid)
                .unwrap_or_else(|| panic!("missing join diff for gate {}", gid))
                .clone();
            // diff = X_in0 - X_in1, so label_in0 = label_in1 + diff
            // and label_in1 = label_in0 - diff.
            if let Some(a) = &lin0 {
                try_set(labels, g.in1, label::sub(a, &diff), queue, system);
            }
            if let Some(b) = &lin1 {
                try_set(labels, g.in0, label::add(b, &diff), queue, system);
            }
        }
        GateType::SameWire => {
            if let Some(a) = &lin0 {
                try_set(labels, g.in1, a.clone(), queue, system);
            }
            if let Some(b) = &lin1 {
                try_set(labels, g.in0, b.clone(), queue, system);
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
