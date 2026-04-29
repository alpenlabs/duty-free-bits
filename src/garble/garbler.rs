//! Garbler: build masks, encode program.
//!
//! Structure mirrors `Exec`: seed constants + inputs, then a worklist fires
//! gates as mask inputs become available. Mask rules per gate:
//!
//! * `Add`/`Sub`: bidirectional — any two of (in0, in1, out) determine the third.
//! * `Mul(s, x)`: the output mask is `s · X_x`. When `s = 0` this is `0`
//!   regardless of `X_x`, so the gate can fire without its input being masked.
//!   This isn't a shortcut — it's the gate's defining equation. It matters
//!   because the last iteration of `word_to_hot_with_bits` is exactly
//!   `Mul(0, acc[0])`, and that's how the circular phantom-bit construction
//!   gets its first mask.
//! * `Mod2k` / `Div2k`: forward only (non-invertible at the mask level).
//! * `Switch`: bidirectional (ctrl+data → out, ctrl+out → data) using
//!   `Y = H(X_ctrl, gid) + X_data`. Garbler does not branch on ctrl value —
//!   masks propagate unconditionally.
//! * `Join`: does not propagate masks (both sides are independently determined).
//! * `SameWire`: symmetric copy.
//!
//! Δ is sampled with low bit forced to 1 (point-and-permute); the evaluator
//! uses it to recover ctrl values from label LSBs.
//!
//! A final pass over gates in creation order emits per-gate material
//! (switch LSBs, join diffs) and appends the declared output masks.

use super::hash;
use super::label::{self, Label, NcfLabel};
use super::program::Program;
use crate::system::System;
use crate::types::{GateId, GateType, Wire};

/// Ensure δ's low bit is 1 so the point-and-permute trick works.
pub fn normalize_delta(delta: u128) -> u128 {
    delta | 1
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
    assert_eq!(delta & 1, 1, "Δ must have low bit 1 (call normalize_delta)");

    let mut masks: Vec<Option<Label>> = vec![None; system.num_wires()];

    // Seed 1: constants. Mask = -c·Δ_R (CF) or -c (NCF).
    for (wid, slot) in masks.iter_mut().enumerate() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        let modulus = v.modulus;
        let neg_c = if v.v == 0 { 0 } else { modulus - v.v };
        *slot = Some(if system.is_cf_flags[wid] {
            let d = label::delta_r(delta, modulus);
            label::scalar_mul(neg_c, &Label::Cf(d))
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
        assert_eq!(m.modulus(), system.modulus(w), "input mask modulus mismatch");
        assert!(m.is_cf(), "input mask must be CF");
        assert!(
            masks[w.wid].is_none(),
            "input wire {} already masked",
            w.wid
        );
        masks[w.wid] = Some(m.clone());
    }

    // Worklist: seed with every gate, then let `try_set` cascades take over.
    // We seed all gates (not just subscribers of seeded wires) because
    // `Mul(0, _)` doesn't depend on its input — its output mask is `0`
    // unconditionally — so it needs a firing chance even if nothing ever
    // updates its input. That's what lets the phantom `bs[i]` wires in
    // `word_to_hot_with_bits` get their first mask.
    let mut queue: Vec<GateId> = (0..system.num_gates()).collect();
    while let Some(gid) = queue.pop() {
        propagate_gate(system, gid, &mut masks, &mut queue);
    }

    // Second pass: emit per-gate material and output masks in deterministic order.
    let mut program = Program::with_num_gates(system.num_gates());
    for (gid, g) in system.gates.iter().enumerate() {
        match g.typ {
            GateType::Switch => {
                let ctrl = masks[g.in1.wid]
                    .as_ref()
                    .unwrap_or_else(|| panic!("switch gate {}: ctrl mask unresolved", gid));
                program.set_switch_lsb(gid, cf_lsb(ctrl));
            }
            GateType::Join => {
                let a = masks[g.in0.wid]
                    .as_ref()
                    .unwrap_or_else(|| panic!("join gate {}: in0 mask unresolved", gid));
                let b = masks[g.in1.wid]
                    .as_ref()
                    .unwrap_or_else(|| panic!("join gate {}: in1 mask unresolved", gid));
                program.set_join_diff(gid, label::sub(a, b));
            }
            _ => {}
        }
    }

    for &w in output_wires {
        assert!(
            !system.is_cf(w),
            "output wire {} must be NCF (use constant_ncf / input_ncf / alloc_wire_kind)",
            w.wid
        );
        let m = masks[w.wid]
            .clone()
            .unwrap_or_else(|| panic!("output wire {}: mask unresolved", w.wid));
        program.push_output_mask(m);
    }

    program
}

fn propagate_gate(
    system: &System,
    gid: GateId,
    masks: &mut [Option<Label>],
    queue: &mut Vec<GateId>,
) {
    let g = system.gates[gid];
    let min0 = masks[g.in0.wid].clone();
    let min1 = masks[g.in1.wid].clone();
    let mout = match g.typ {
        GateType::Add
        | GateType::Sub
        | GateType::Mul
        | GateType::Mod2k
        | GateType::Div2k
        | GateType::Switch => masks[g.out.wid].clone(),
        _ => None,
    };

    match g.typ {
        GateType::Add => {
            // out = in0 + in1  ⇔  in0 = out - in1  ⇔  in1 = out - in0
            if let (Some(a), Some(b)) = (&min0, &min1) {
                try_set(masks, g.out, label::add(a, b), queue, system);
            }
            if let (Some(o), Some(b)) = (&mout, &min1) {
                try_set(masks, g.in0, label::sub(o, b), queue, system);
            }
            if let (Some(a), Some(o)) = (&min0, &mout) {
                try_set(masks, g.in1, label::sub(o, a), queue, system);
            }
        }
        GateType::Sub => {
            // out = in0 - in1  ⇔  in0 = out + in1  ⇔  in1 = in0 - out
            if let (Some(a), Some(b)) = (&min0, &min1) {
                try_set(masks, g.out, label::sub(a, b), queue, system);
            }
            if let (Some(o), Some(b)) = (&mout, &min1) {
                try_set(masks, g.in0, label::add(o, b), queue, system);
            }
            if let (Some(a), Some(o)) = (&min0, &mout) {
                try_set(masks, g.in1, label::sub(a, o), queue, system);
            }
        }
        GateType::Mul => {
            // X_out = s · X_in. When s = 0 this is 0 regardless of X_in, so the
            // gate is fireable without its input being masked.
            if g.param == 0 {
                try_set(masks, g.out, Label::zero(system.is_cf(g.out), system.modulus(g.out)), queue, system);
            } else if let Some(a) = &min0 {
                try_set(masks, g.out, label::scalar_mul(g.param, a), queue, system);
            }
        }
        GateType::Mod2k => {
            if let Some(a) = &min0 {
                try_set(masks, g.out, label::mod2k(a, g.param as u32), queue, system);
            }
        }
        GateType::Div2k => {
            if let Some(a) = &min0 {
                try_set(masks, g.out, label::div2k(a, g.param as u32), queue, system);
            }
        }
        GateType::Switch => {
            // out = H(ctrl, gid) + data  ⇔  data = out - H(ctrl, gid)
            if let Some(ctrl) = &min1 {
                let h = hash::hash(ctrl, gid, system.is_cf(g.out), system.modulus(g.out));
                if let Some(data) = &min0 {
                    try_set(masks, g.out, label::add(&h, data), queue, system);
                }
                if let Some(o) = &mout {
                    try_set(masks, g.in0, label::sub(o, &h), queue, system);
                }
            }
        }
        GateType::Join => {
            // No mask propagation; join diff is emitted in the final pass.
        }
        GateType::SameWire => {
            if let Some(a) = &min0 {
                try_set(masks, g.in1, a.clone(), queue, system);
            }
            if let Some(b) = &min1 {
                try_set(masks, g.in0, b.clone(), queue, system);
            }
        }
    }
}

fn try_set(
    masks: &mut [Option<Label>],
    w: Wire,
    new_mask: Label,
    queue: &mut Vec<GateId>,
    system: &System,
) {
    if masks[w.wid].is_none() {
        assert_eq!(
            new_mask.modulus(),
            system.modulus(w),
            "modulus mismatch setting wire {}",
            w.wid
        );
        masks[w.wid] = Some(new_mask);
        queue.extend_from_slice(&system.subscriptions[w.wid]);
    }
}

/// Low bit of a CF label's coord 0 (for point-and-permute).
fn cf_lsb(l: &Label) -> bool {
    match l {
        Label::Cf(c) => (c.get(0) & 1) == 1,
        Label::Ncf(_) => panic!("cf_lsb: NCF label"),
    }
}
