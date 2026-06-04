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
//! Δ is the global Free-XOR offset; its low bit is forced to 1 so that `Δ_R(2)`
//! is nonzero and CF Z_2 labels distinguish 0 from 1.
//!
//! A final pass over gates in creation order emits the join diffs (the only
//! switch-system communication — switches reveal nothing) and appends the
//! declared output masks.

use crate::hash;
use crate::label::{self, Label, NcfLabel};
use super::program::Program;
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

/// Force Δ's low bit to 1, so `Δ_R(2)` is nonzero (CF Z_2 labels must separate
/// value 0 from value 1).
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
    let mut bulk_cache: BulkCache = vec![None; system.num_switch_groups()];
    let mut queue: Vec<GateId> = (0..system.num_gates()).collect();
    while let Some(gid) = queue.pop() {
        propagate_gate(system, gid, &mut masks, &mut queue, &mut bulk_cache);
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

fn propagate_gate(
    system: &System,
    gid: GateId,
    masks: &mut [Option<Label>],
    queue: &mut Vec<GateId>,
    bulk_cache: &mut BulkCache,
) {
    match system.gates[gid] {
        Gate::Add { in0, in1, out } => {
            // out = in0 + in1  ⇔  in0 = out - in1  ⇔  in1 = out - in0
            let (m0, m1, mo) = (masks[in0.wid].clone(), masks[in1.wid].clone(), masks[out.wid].clone());
            if let (Some(a), Some(b)) = (&m0, &m1) {
                try_set(masks, out, label::add(a, b), queue, system);
            }
            if let (Some(o), Some(b)) = (&mo, &m1) {
                try_set(masks, in0, label::sub(o, b), queue, system);
            }
            if let (Some(a), Some(o)) = (&m0, &mo) {
                try_set(masks, in1, label::sub(o, a), queue, system);
            }
        }
        Gate::Sub { in0, in1, out } => {
            // out = in0 - in1  ⇔  in0 = out + in1  ⇔  in1 = in0 - out
            let (m0, m1, mo) = (masks[in0.wid].clone(), masks[in1.wid].clone(), masks[out.wid].clone());
            if let (Some(a), Some(b)) = (&m0, &m1) {
                try_set(masks, out, label::sub(a, b), queue, system);
            }
            if let (Some(o), Some(b)) = (&mo, &m1) {
                try_set(masks, in0, label::add(o, b), queue, system);
            }
            if let (Some(a), Some(o)) = (&m0, &mo) {
                try_set(masks, in1, label::sub(a, o), queue, system);
            }
        }
        Gate::Mul { in0, scalar, out } => {
            // X_out = s · X_in. When s = 0 this is 0 regardless of X_in, so the
            // gate is fireable without its input being masked.
            if scalar == 0 {
                try_set(masks, out, Label::zero(system.is_cf(out), system.modulus(out)), queue, system);
            } else if let Some(a) = masks[in0.wid].clone() {
                try_set(masks, out, label::scalar_mul(scalar, &a), queue, system);
            }
        }
        Gate::Mod2k { in0, k, out } => {
            if let Some(a) = masks[in0.wid].clone() {
                try_set(masks, out, label::mod2k(&a, k), queue, system);
            }
        }
        Gate::Div2k { in0, k, out } => {
            if let Some(a) = masks[in0.wid].clone() {
                try_set(masks, out, label::div2k(&a, k), queue, system);
            }
        }
        Gate::Switch { data, ctrl, out } => {
            // out = H(ctrl, gid) + data  ⇔  data = out - H(ctrl, gid).
            // For grouped switches, H is sliced from a single wide bulk call.
            if let Some(c) = masks[ctrl.wid].clone() {
                let h = switch_hash(system, gid, &c, bulk_cache);
                if let Some(d) = masks[data.wid].clone() {
                    try_set(masks, out, label::add(&h, &d), queue, system);
                }
                if let Some(o) = masks[out.wid].clone() {
                    try_set(masks, data, label::sub(&o, &h), queue, system);
                }
            }
        }
        Gate::Join { .. } => {
            // No mask propagation; the join diff is emitted in the final pass.
        }
        Gate::SameWire { a, b } => {
            if let Some(ma) = masks[a.wid].clone() {
                try_set(masks, b, ma, queue, system);
            }
            if let Some(mb) = masks[b.wid].clone() {
                try_set(masks, a, mb, queue, system);
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
