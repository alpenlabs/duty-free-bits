//! Evaluator: derive output labels by replaying a cleartext-execution journal.
//!
//! Each wire's evaluator label takes the form `X + x·Δ_R` (CF) or `X + x`
//! (NCF), where `X` is the garbler's mask. The evaluator knows the cleartext
//! `x`, so it first runs [`Exec`](crate::exec::Exec) to learn every wire's
//! value (and every switch control), then [`replay_with_labels`] walks that
//! journal, deriving each label through the same gate direction the value
//! pass took — forward through affine gates, backward too, and through a
//! switch only when its control is 0.

use super::program::Program;
use crate::exec::JournalEntry;
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
    nonce_base: u64,
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
        hash::hash_solo(
            ctrl_label,
            nonce_base + gid as u64,
            system.is_cf(out),
            system.modulus(out),
        )
    }
}

/// Evaluate a garbled system by replaying a cleartext-execution journal.
///
/// `journal` is the set-order record of an [`Exec::run_recorded`] pass
/// (`crate::exec`): step n says gate `gid` derived wire `wid`. Label
/// propagation derives wires through exactly the same gate directions as value
/// propagation — joins via the program diff, switches only when the cleartext
/// control is 0 — so by induction every operand a step reads is label-known
/// when the step executes, and each label is the unique value consistent with
/// the garbled program. The journal is a linear tape: no queue, no wakeups, no
/// definedness checks.
///
/// [`Exec::run_recorded`]: crate::exec::Exec::run_recorded
#[allow(clippy::too_many_arguments)]
pub fn replay_with_labels(
    system: &System,
    input_wires: &[Wire],
    input_labels: &[Label],
    values: &[Val],
    program: &Program,
    journal: &[JournalEntry],
    output_wires: &[Wire],
    nonce_base: u64,
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
        // Every arm verifies the journaled wid is actually a wire of gate gid —
        // a journal from a different Exec/System must fail loudly, never write
        // to an unrelated wire.
        let bad_wid = || -> ! {
            panic!(
                "replay: journal pairs gate {gid} with wire {wid}, which the \
                 gate does not touch — journal/system mismatch"
            )
        };
        let v = match system.gates[gid] {
            Gate::Add { in0, in1, out } => {
                if wid == out.wid {
                    label::add(lbl(in0), lbl(in1)) // out = in0 + in1
                } else if wid == in0.wid {
                    label::sub(lbl(out), lbl(in1)) // in0 = out - in1
                } else if wid == in1.wid {
                    label::sub(lbl(out), lbl(in0)) // in1 = out - in0
                } else {
                    bad_wid()
                }
            }
            Gate::Sub { in0, in1, out } => {
                if wid == out.wid {
                    label::sub(lbl(in0), lbl(in1)) // out = in0 - in1
                } else if wid == in0.wid {
                    label::add(lbl(out), lbl(in1)) // in0 = out + in1
                } else if wid == in1.wid {
                    label::sub(lbl(in0), lbl(out)) // in1 = in0 - out
                } else {
                    bad_wid()
                }
            }
            Gate::Mul { in0, scalar, out } => {
                if wid != out.wid {
                    bad_wid()
                }
                if scalar == 0 {
                    Label::zero(system.is_cf(out), system.modulus(out))
                } else {
                    label::scalar_mul(scalar, lbl(in0))
                }
            }
            Gate::Mod2k { in0, k, out } => {
                if wid != out.wid {
                    bad_wid()
                }
                label::mod2k(lbl(in0), k)
            }
            Gate::Div2k { in0, k, out } => {
                if wid != out.wid {
                    bad_wid()
                }
                label::div2k(lbl(in0), k)
            }
            Gate::Switch { data, ctrl, out } => {
                // Exec fires a switch only when its control value is 0; a
                // journal from a different Exec/input would silently produce
                // garbage labels here, so check unconditionally.
                if wid != out.wid && wid != data.wid {
                    bad_wid()
                }
                assert_eq!(values[ctrl.wid].v, 0, "replay: switch {gid} open");
                let h = switch_hash(system, gid, lbl(ctrl), &mut bulk_cache, nonce_base);
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
                } else if wid == aw.wid {
                    label::add(lbl(bw), diff)
                } else {
                    bad_wid()
                }
            }
            Gate::SameWire { a: aw, b: bw } => {
                if wid == bw.wid {
                    lbl(aw).clone()
                } else if wid == aw.wid {
                    lbl(bw).clone()
                } else {
                    bad_wid()
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
