//! Cleartext execution of a [`System`] — the ground-truth semantics.
//!
//! `Exec` propagates concrete wire values through the gates (forward and
//! backward) to fixpoint. The evaluator runs it on its known input `x` to read switch controls in cleartext.

use crate::system::System;
use crate::types::*;

/// Worklist state shared by the three fixpoint engines ([`Exec`],
/// `comp_gc::garbler`, `comp_gc::evaluator`).
///
/// Invariants:
/// * `known[w]` ⇔ the engine's slot for wire `w` is determined. Definedness
///   checks read this bitset; the wide per-wire slot arrays are loaded only to
///   fetch operands for an actual computation.
/// * `done[g]` ⇒ gate `g` can never derive a new wire. Done gates are dropped
///   at pop and never (re-)enqueued.
#[derive(Debug)]
pub(crate) struct Worklist {
    /// One bit per wire.
    known: Vec<u64>,
    /// One bit per gate.
    done: Vec<u64>,
    /// One bit per gate: currently in `queue` (dedups pushes — a gate woken by
    /// several wire updates before it fires need only be visited once).
    queued: Vec<u64>,
    /// LIFO gate worklist.
    queue: Vec<GateId>,
}

impl Worklist {
    pub(crate) fn new(num_wires: usize, num_gates: usize) -> Self {
        Worklist {
            known: vec![0; num_wires.div_ceil(64)],
            done: vec![0; num_gates.div_ceil(64)],
            queued: vec![0; num_gates.div_ceil(64)],
            queue: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn wire_known(&self, wid: usize) -> bool {
        (self.known[wid >> 6] >> (wid & 63)) & 1 != 0
    }

    #[inline]
    pub(crate) fn mark_known(&mut self, wid: usize) {
        self.known[wid >> 6] |= 1u64 << (wid & 63);
    }

    #[inline]
    pub(crate) fn gate_done(&self, gid: GateId) -> bool {
        (self.done[gid >> 6] >> (gid & 63)) & 1 != 0
    }

    #[inline]
    pub(crate) fn mark_done(&mut self, gid: GateId) {
        self.done[gid >> 6] |= 1u64 << (gid & 63);
    }

    /// Seed the queue with one gate (done gates are still dropped at pop).
    #[inline]
    pub(crate) fn enqueue(&mut self, gid: GateId) {
        if (self.queued[gid >> 6] >> (gid & 63)) & 1 == 0 {
            self.queued[gid >> 6] |= 1u64 << (gid & 63);
            self.queue.push(gid);
        }
    }

    /// Seed the queue with a subscription list.
    #[inline]
    pub(crate) fn enqueue_all(&mut self, gids: &[u32]) {
        for &g in gids {
            self.enqueue(g as GateId);
        }
    }

    /// Pop the next live gate, dropping done ones.
    #[inline]
    pub(crate) fn pop_live(&mut self) -> Option<GateId> {
        while let Some(gid) = self.queue.pop() {
            self.queued[gid >> 6] &= !(1u64 << (gid & 63));
            if !self.gate_done(gid) {
                return Some(gid);
            }
        }
        None
    }

    /// Wake `w`'s subscribers after `src` set it — except `src` itself (its
    /// firing pass already sees the new wire) and except done or
    /// already-queued gates.
    #[inline]
    pub(crate) fn wake_subscribers(&mut self, subs: &[u32], src: GateId) {
        for &g in subs {
            let g = g as GateId;
            if g != src && !self.gate_done(g) {
                self.enqueue(g);
            }
        }
    }
}

/// One step of a value-propagation run: gate `gid` derived wire `wid`.
///
/// A recorded run is a valid *label* schedule for the evaluator: label
/// propagation derives wires through exactly the same gate directions as
/// value propagation (joins via the diff, switches only when the cleartext
/// control is 0), so by induction every operand of step n is label-known
/// after steps 0..n — see `comp_gc::evaluator::replay_with_labels`.
#[derive(Clone, Copy, Debug)]
pub struct JournalEntry {
    /// Gate that fired.
    pub gid: u32,
    /// Wire it derived.
    pub wid: u32,
}

/// Concrete executor: propagates wire values by eagerly firing gates.
#[derive(Debug)]
pub struct Exec<'a> {
    system: &'a System,
    values: Vec<Val>,
    /// Set-order journal, recorded only by [`run_recorded`](Exec::run_recorded).
    journal: Vec<JournalEntry>,
    record: bool,
}

impl<'a> Exec<'a> {
    /// Create an executor seeded with the system's initial values.
    pub fn new(system: &'a System) -> Self {
        Exec {
            system,
            values: system.values.clone(),
            journal: Vec::new(),
            record: false,
        }
    }

    /// Read the current value of a wire.
    pub fn get(&self, w: Wire) -> Val {
        self.values[w.wid]
    }

    /// All wire values (indexed by wire id).
    pub fn values(&self) -> &[Val] {
        &self.values
    }

    /// Set a wire's value (must be currently undefined).
    pub fn set(&mut self, w: Wire, v: Val) {
        assert!(self.values[w.wid].is_none(), "wire {} already set", w.wid);
        assert_eq!(self.values[w.wid].modulus, v.modulus);
        self.values[w.wid] = v;
    }

    /// Propagate all known values until no more values change.
    pub fn run(&mut self) {
        // The worklist is rebuilt per call: values `set` between runs must
        // land in `known`.
        let mut wl = Worklist::new(self.values.len(), self.system.num_gates());
        let subs = self.system.subscriptions();
        for wid in 0..self.values.len() {
            if !self.values[wid].is_none() {
                wl.mark_known(wid);
                wl.enqueue_all(subs.of(wid));
            }
        }
        self.propagate(&mut wl);
    }

    /// [`run`](Exec::run), additionally recording the set-order journal
    /// (readable via [`journal`](Exec::journal)).
    ///
    /// Seeds (`set` calls and constants) are not journaled — only derived
    /// wires, in derivation order.
    pub fn run_recorded(&mut self) {
        self.record = true;
        self.run();
        self.record = false;
    }

    /// The set-order journal of the last [`run_recorded`](Exec::run_recorded).
    pub fn journal(&self) -> &[JournalEntry] {
        &self.journal
    }

    fn try_set(&mut self, w: Wire, v: Val, src: GateId, wl: &mut Worklist) {
        if !wl.wire_known(w.wid) {
            debug_assert!(!v.is_none());
            assert_eq!(
                self.values[w.wid].modulus, v.modulus,
                "modulus mismatch on wire {}: expected {}, got {}",
                w.wid, self.values[w.wid].modulus, v.modulus
            );
            self.values[w.wid] = v;
            if self.record {
                self.journal.push(JournalEntry {
                    gid: src as u32,
                    wid: w.wid as u32,
                });
            }
            wl.mark_known(w.wid);
            wl.wake_subscribers(self.system.subscriptions().of(w.wid), src);
        }
    }

    fn propagate(&mut self, wl: &mut Worklist) {
        while let Some(gid) = wl.pop_live() {
            let gate_done = match self.system.gates[gid] {
                Gate::Switch { data, ctrl, out } => {
                    // The switch constrains nothing until its control value is
                    // defined; `values` is fixed mid-run, so a defined non-zero
                    // control means the switch is open forever.
                    if !wl.wire_known(ctrl.wid) {
                        false
                    } else if self.values[ctrl.wid].v != 0 {
                        true // open switch: never propagates
                    } else {
                        // Forward: if ctrl=0 and data known, out = data.
                        if !wl.wire_known(out.wid) && wl.wire_known(data.wid) {
                            self.try_set(out, self.values[data.wid], gid, wl);
                        }
                        // Backward: if ctrl=0 and out known, data = out.
                        if !wl.wire_known(data.wid) && wl.wire_known(out.wid) {
                            self.try_set(data, self.values[out.wid], gid, wl);
                        }
                        wl.wire_known(data.wid) && wl.wire_known(out.wid)
                    }
                }
                Gate::Join { a, b } | Gate::SameWire { a, b } => {
                    if !wl.wire_known(a.wid) && wl.wire_known(b.wid) {
                        self.try_set(a, self.values[b.wid], gid, wl);
                    }
                    if !wl.wire_known(b.wid) && wl.wire_known(a.wid) {
                        self.try_set(b, self.values[a.wid], gid, wl);
                    }
                    wl.wire_known(a.wid) && wl.wire_known(b.wid)
                }
                Gate::Add { in0, in1, out } => {
                    if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                        let v = val_add(self.values[in0.wid], self.values[in1.wid]);
                        self.try_set(out, v, gid, wl); // out = in0 + in1
                    }
                    if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                        let v = val_sub(self.values[out.wid], self.values[in1.wid]);
                        self.try_set(in0, v, gid, wl); // in0 = out - in1
                    }
                    if !wl.wire_known(in1.wid) && wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                        let v = val_sub(self.values[out.wid], self.values[in0.wid]);
                        self.try_set(in1, v, gid, wl); // in1 = out - in0
                    }
                    wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
                }
                Gate::Sub { in0, in1, out } => {
                    if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) && wl.wire_known(in1.wid) {
                        let v = val_sub(self.values[in0.wid], self.values[in1.wid]);
                        self.try_set(out, v, gid, wl); // out = in0 - in1
                    }
                    if !wl.wire_known(in0.wid) && wl.wire_known(out.wid) && wl.wire_known(in1.wid) {
                        let v = val_add(self.values[out.wid], self.values[in1.wid]);
                        self.try_set(in0, v, gid, wl); // in0 = out + in1
                    }
                    if !wl.wire_known(in1.wid) && wl.wire_known(in0.wid) && wl.wire_known(out.wid) {
                        let v = val_sub(self.values[in0.wid], self.values[out.wid]);
                        self.try_set(in1, v, gid, wl); // in1 = in0 - out
                    }
                    wl.wire_known(in0.wid) && wl.wire_known(in1.wid) && wl.wire_known(out.wid)
                }
                Gate::Mul { in0, scalar, out } => {
                    if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                        let v = val_mul(scalar, self.values[in0.wid]);
                        self.try_set(out, v, gid, wl);
                    }
                    wl.wire_known(out.wid)
                }
                Gate::Mod2k { in0, k, out } => {
                    if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                        let v = val_mod2k(self.values[in0.wid], k);
                        self.try_set(out, v, gid, wl);
                    }
                    wl.wire_known(out.wid)
                }
                Gate::Div2k { in0, k, out } => {
                    if !wl.wire_known(out.wid) && wl.wire_known(in0.wid) {
                        let v = val_div2k(self.values[in0.wid], k);
                        self.try_set(out, v, gid, wl);
                    }
                    wl.wire_known(out.wid)
                }
            };
            if gate_done {
                wl.mark_done(gid);
            }
        }
    }
}
