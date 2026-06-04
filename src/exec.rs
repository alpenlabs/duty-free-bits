use crate::system::System;
use crate::types::*;

/// Concrete executor: propagates wire values by eagerly firing gates.
#[derive(Debug)]
pub struct Exec<'a> {
    system: &'a System,
    values: Vec<Val>,
}

impl<'a> Exec<'a> {
    /// Create an executor seeded with the system's initial values.
    pub fn new(system: &'a System) -> Self {
        Exec {
            system,
            values: system.values.clone(),
        }
    }

    /// Read the current value of a wire.
    pub fn get(&self, w: Wire) -> Val {
        self.values[w.wid]
    }

    /// All wire values (indexed by wire id) — used by the evaluator to read
    /// switch controls in cleartext.
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
        let mut queue = Vec::new();
        for wid in 0..self.values.len() {
            if !self.values[wid].is_none() {
                queue.extend_from_slice(&self.system.subscriptions[wid]);
            }
        }
        self.propagate(&mut queue);
    }

    fn try_set(&mut self, w: Wire, v: Val, queue: &mut Vec<GateId>) {
        if self.values[w.wid].is_none() && !v.is_none() {
            assert_eq!(
                self.values[w.wid].modulus, v.modulus,
                "modulus mismatch on wire {}: expected {}, got {}",
                w.wid, self.values[w.wid].modulus, v.modulus
            );
            self.values[w.wid] = v;
            queue.extend_from_slice(&self.system.subscriptions[w.wid]);
        }
    }

    fn propagate(&mut self, queue: &mut Vec<GateId>) {
        while let Some(gid) = queue.pop() {
            match self.system.gates[gid] {
                Gate::Switch { data, ctrl, out } => {
                    let (vdata, vctrl, vout) = (self.get(data), self.get(ctrl), self.get(out));
                    // Forward: if ctrl=0 and data known, out = data.
                    self.try_set(out, guard(vdata, vctrl), queue);
                    // Backward: if ctrl=0 and out known, data = out.
                    self.try_set(data, guard(vout, vctrl), queue);
                }
                Gate::Join { a, b } | Gate::SameWire { a, b } => {
                    let (va, vb) = (self.get(a), self.get(b));
                    self.try_set(a, vb, queue);
                    self.try_set(b, va, queue);
                }
                Gate::Add { in0, in1, out } => {
                    let (v0, v1, vo) = (self.get(in0), self.get(in1), self.get(out));
                    self.try_set(out, val_add(v0, v1), queue); // out = in0 + in1
                    self.try_set(in0, val_sub(vo, v1), queue); // in0 = out - in1
                    self.try_set(in1, val_sub(vo, v0), queue); // in1 = out - in0
                }
                Gate::Sub { in0, in1, out } => {
                    let (v0, v1, vo) = (self.get(in0), self.get(in1), self.get(out));
                    self.try_set(out, val_sub(v0, v1), queue); // out = in0 - in1
                    self.try_set(in0, val_add(vo, v1), queue); // in0 = out + in1
                    self.try_set(in1, val_sub(v0, vo), queue); // in1 = in0 - out
                }
                Gate::Mul { in0, scalar, out } => {
                    self.try_set(out, val_mul(scalar, self.get(in0)), queue);
                }
                Gate::Mod2k { in0, k, out } => {
                    self.try_set(out, val_mod2k(self.get(in0), k), queue);
                }
                Gate::Div2k { in0, k, out } => {
                    self.try_set(out, val_div2k(self.get(in0), k), queue);
                }
            }
        }
    }
}
