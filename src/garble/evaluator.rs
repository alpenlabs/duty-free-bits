//! Evaluator: decode labels using a worklist, mirroring `Exec`'s structure.
//!
//! Each wire's evaluator label ultimately takes the form `X + x·Δ_R` (CF) or
//! `X + x` (NCF), where `X` is the garbler's mask. Labels are derived in *any*
//! order the graph permits — forward through affine gates, backward too, and
//! through switches only when the ctrl LSB tells us `ctrl = 0`. This is what
//! allows patterns like `ohe_scale`'s "junk sh_i labels, join fixes the sum,
//! backward add recovers individual sh_i" to work.

use super::hash;
use super::label::{self, Label};
use super::program::{GateMaterial, Program};
use crate::system::System;
use crate::types::{GateId, GateType, Val, Wire};

/// Evaluate a garbled system.
pub fn eval(
    system: &System,
    input_wires: &[Wire],
    input_values: &[Val],
    delta: u128,
    input_masks: &[Label],
    program: &Program,
    output_wires: &[Wire],
) -> Vec<Val> {
    assert_eq!(input_wires.len(), input_values.len());
    assert_eq!(input_wires.len(), input_masks.len());
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

    // Inputs: label = mask + value · Δ_R.
    for ((&w, &val), mask) in input_wires
        .iter()
        .zip(input_values.iter())
        .zip(input_masks.iter())
    {
        assert_eq!(val.modulus, system.modulus(w));
        assert!(val.defined);
        let d = label::delta_r(delta, val.modulus);
        let contrib = label::scalar_mul(val.v, &Label::Cf(d));
        labels[w.wid] = Some(label::add(mask, &contrib));
    }

    // Seed worklist: every gate subscribed to a currently-labeled wire.
    let mut queue: Vec<GateId> = Vec::new();
    for (wid, slot) in labels.iter().enumerate() {
        if slot.is_some() {
            queue.extend_from_slice(&system.subscriptions[wid]);
        }
    }

    // Propagate.
    while let Some(gid) = queue.pop() {
        fire_gate(system, gid, &mut labels, program, &mut queue);
    }

    // Decode outputs.
    let mut out = Vec::with_capacity(output_wires.len());
    for (i, &w) in output_wires.iter().enumerate() {
        let out_mask = &program.output_masks()[i];
        let lbl = labels[w.wid]
            .clone()
            .unwrap_or_else(|| panic!("no label on output wire {}", w.wid));
        let stripped = label::sub(&lbl, out_mask);
        match stripped {
            Label::Ncf(n) => out.push(Val::new(n.rep, n.modulus)),
            Label::Cf(_) => panic!("output wire {} must be NCF", w.wid),
        }
    }
    out
}

fn fire_gate(
    system: &System,
    gid: GateId,
    labels: &mut [Option<Label>],
    program: &Program,
    queue: &mut Vec<GateId>,
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
            // Need ctrl label to decrypt ctrl, plus ctrl lsb from program.
            let Some(ctrl_label) = lin1 else { return; };
            let lsb_garbler = match program.material(gid) {
                Some(GateMaterial::SwitchLsb(b)) => *b,
                _ => panic!("missing switch lsb for gate {}", gid),
            };
            let ctrl_value = match &ctrl_label {
                Label::Cf(c) => {
                    assert_eq!(c.modulus(), 2, "switch ctrl must be Z_2");
                    let lsb_eval = (c.get(0) & 1) == 1;
                    (lsb_eval ^ lsb_garbler) as u64
                }
                Label::Ncf(_) => panic!("switch ctrl must be CF"),
            };
            if ctrl_value != 0 {
                // Switch does not fire: no label propagation.
                return;
            }
            // ctrl = 0: propagate in either direction via H(ctrl_label, gid).
            let h = hash::hash(&ctrl_label, gid, system.is_cf(g.out), system.modulus(g.out));
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
            let diff = match program.material(gid) {
                Some(GateMaterial::JoinDiff(d)) => d.clone(),
                _ => panic!("missing join diff for gate {}", gid),
            };
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
