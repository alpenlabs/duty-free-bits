use crate::types::*;
use crate::system::System;

/// Concrete executor: propagates wire values by eagerly firing gates.
#[derive(Debug)]
pub struct Exec<'a> {
    system: &'a System,
    values: Vec<Val>,
}

impl<'a> Exec<'a> {
    /// Create an executor seeded with the system's initial values.
    pub fn new(system: &'a System) -> Self {
        Exec { system, values: system.values.clone() }
    }

    /// Read the current value of a wire.
    pub fn get(&self, w: Wire) -> Val {
        self.values[w.wid]
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
            assert_eq!(self.values[w.wid].modulus, v.modulus,
                "modulus mismatch on wire {}: expected {}, got {}",
                w.wid, self.values[w.wid].modulus, v.modulus);
            self.values[w.wid] = v;
            queue.extend_from_slice(&self.system.subscriptions[w.wid]);
        }
    }

    fn propagate(&mut self, queue: &mut Vec<GateId>) {
        while let Some(gid) = queue.pop() {
            let g = self.system.gates[gid];
            let vin0 = self.get(g.in0);
            let vin1 = self.get(g.in1);
            let vout = self.get(g.out);

            match g.typ {
                GateType::Switch => {
                    // Forward: if ctrl=0 and data known, out = data
                    self.try_set(g.out, guard(vin0, vin1), queue);
                    // Backward: if ctrl=0 and out known, data = out
                    self.try_set(g.in0, guard(vout, vin1), queue);
                }
                GateType::Join => {
                    self.try_set(g.in0, vin1, queue);
                    self.try_set(g.in1, vin0, queue);
                }
                GateType::SameWire => {
                    self.try_set(g.in0, vin1, queue);
                    self.try_set(g.in1, vin0, queue);
                }
                GateType::Add => {
                    // out = in0 + in1
                    self.try_set(g.out, val_add(vin0, vin1), queue);
                    // in0 = out - in1
                    self.try_set(g.in0, val_sub(vout, vin1), queue);
                    // in1 = out - in0
                    self.try_set(g.in1, val_sub(vout, vin0), queue);
                }
                GateType::Sub => {
                    // out = in0 - in1
                    self.try_set(g.out, val_sub(vin0, vin1), queue);
                    // in0 = out + in1
                    self.try_set(g.in0, val_add(vout, vin1), queue);
                    // in1 = in0 - out
                    self.try_set(g.in1, val_sub(vin0, vout), queue);
                }
                GateType::Mul => {
                    self.try_set(g.out, val_mul(g.param, vin0), queue);
                }
                GateType::Mod2k => {
                    self.try_set(g.out, val_mod2k(vin0, g.param as u32), queue);
                }
                GateType::Div2k => {
                    self.try_set(g.out, val_div2k(vin0, g.param as u32), queue);
                }
            }
        }
    }
}
