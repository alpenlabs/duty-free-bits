use crate::types::*;

/// The constraint system: holds wires, gates, and propagation queue.
#[derive(Debug)]
pub struct System {
    pub(crate) gates: Vec<Gate>,
    pub(crate) values: Vec<Val>,
    pub(crate) subscriptions: Vec<Vec<GateId>>,
    /// Total join cost in bits accumulated so far.
    pub join_complexity: usize,
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}

impl System {
    /// Create a new, empty system.
    pub fn new() -> Self {
        System {
            gates: Vec::new(),
            values: Vec::new(),
            subscriptions: Vec::new(),
            join_complexity: 0,
        }
    }

    /// Allocate a fresh wire in Z_modulus (initially undefined).
    pub fn alloc_wire(&mut self, modulus: u64) -> Wire {
        let wid = self.values.len();
        self.subscriptions.push(Vec::new());
        self.values.push(Val::none(modulus));
        Wire { wid }
    }

    fn subscribe(&mut self, w: Wire, gid: GateId) {
        self.subscriptions[w.wid].push(gid);
    }

    /// Get the modulus for a wire.
    pub fn modulus(&self, x: Wire) -> u64 {
        self.values[x.wid].modulus
    }

    /// Get bitlen for a wire in Z_{2^k}. Panics if modulus is not a power of 2.
    pub fn bitlen(&self, x: Wire) -> u32 {
        let m = self.modulus(x);
        assert!(
            m.is_power_of_two(),
            "bitlen: modulus {} is not a power of 2",
            m
        );
        m.ilog2()
    }

    /// Number of wires in the system.
    pub fn num_wires(&self) -> usize {
        self.values.len()
    }

    /// Number of gates in the system.
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    // --- Wire constructors ---

    /// Create a fresh input wire in Z_modulus.
    pub fn input(&mut self, modulus: u64) -> Wire {
        self.alloc_wire(modulus)
    }

    /// Convenience: input wire in Z_{2^bl}
    pub fn input_bits(&mut self, bl: u32) -> Wire {
        self.input(1u64 << bl)
    }

    /// Create a constant wire holding `n` in Z_modulus.
    pub fn constant(&mut self, n: u64, modulus: u64) -> Wire {
        let w = self.alloc_wire(modulus);
        self.values[w.wid] = Val::new(n % modulus, modulus);
        w
    }

    /// Convenience: constant wire in Z_{2^bl}
    pub fn constant_bits(&mut self, n: u64, bl: u32) -> Wire {
        self.constant(n, 1u64 << bl)
    }

    // --- Gate management ---

    fn add_gate(&mut self, g: Gate) {
        let gid = self.gates.len();
        self.gates.push(g);

        self.subscribe(g.in0, gid);
        match g.typ {
            GateType::Switch | GateType::Add | GateType::Sub => {
                self.subscribe(g.in1, gid);
                self.subscribe(g.out, gid);
            }
            GateType::Join | GateType::SameWire => {
                self.subscribe(g.in1, gid);
            }
            GateType::Mul | GateType::Mod2k | GateType::Div2k => {}
        }
    }

    fn one_in_one_out(&mut self, typ: GateType, x: Wire, param: u64, out_mod: u64) -> Wire {
        let out = self.alloc_wire(out_mod);
        let g = Gate {
            typ,
            param,
            in0: x,
            in1: Wire { wid: 0 },
            out,
        };
        self.add_gate(g);
        out
    }

    // --- Core gate constructors ---

    /// Switch: data wire x (any ring), control wire s (must be Z_2).
    /// Output has same modulus as x. If s=0, output=x.
    pub fn switch(&mut self, x: Wire, s: Wire) -> Wire {
        assert_eq!(self.modulus(s), 2, "switch control must be binary (Z_2)");
        let out = self.alloc_wire(self.modulus(x));
        let g = Gate {
            typ: GateType::Switch,
            param: 0,
            in0: x,
            in1: s,
            out,
        };
        self.add_gate(g);
        out
    }

    /// Join: constrain x = y. Both must have the same modulus.
    /// Costs lg|modulus| bits of join complexity.
    pub fn join(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        let m = self.modulus(x);
        // join complexity in bits
        self.join_complexity += (m as f64).log2().ceil() as usize;
        let g = Gate {
            typ: GateType::Join,
            param: 0,
            in0: x,
            in1: y,
            out: Wire { wid: 0 },
        };
        self.add_gate(g);
        x
    }

    /// SameWire: constrain x = y without join cost (when one side is unconstrained).
    pub fn same_wire(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        let g = Gate {
            typ: GateType::SameWire,
            param: 0,
            in0: x,
            in1: y,
            out: Wire { wid: 0 },
        };
        self.add_gate(g);
        x
    }

    /// Addition in the same ring.
    pub fn add(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        let out = self.alloc_wire(self.modulus(x));
        let g = Gate {
            typ: GateType::Add,
            param: 0,
            in0: x,
            in1: y,
            out,
        };
        self.add_gate(g);
        out
    }

    /// Subtraction in the same ring.
    pub fn sub(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        let out = self.alloc_wire(self.modulus(x));
        let g = Gate {
            typ: GateType::Sub,
            param: 0,
            in0: x,
            in1: y,
            out,
        };
        self.add_gate(g);
        out
    }

    /// Scalar multiplication by constant s (mod wire's modulus).
    pub fn mul(&mut self, s: u64, x: Wire) -> Wire {
        self.one_in_one_out(GateType::Mul, x, s, self.modulus(x))
    }

    /// Modular reduction: x mod 2^k. Input must be in Z_{2^n} with k ≤ n.
    pub fn mod2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k <= m.ilog2());
        self.one_in_one_out(GateType::Mod2k, x, k as u64, 1u64 << k)
    }

    /// Division by 2^k. Input must be in Z_{2^{k+c}}, output in Z_{2^c}.
    pub fn div2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k < m.ilog2());
        self.one_in_one_out(GateType::Div2k, x, k as u64, m >> k)
    }

    // --- Derived operations ---

    /// Boolean NOT (Z_2 wire)
    pub fn not(&mut self, x: Wire) -> Wire {
        assert_eq!(self.modulus(x), 2);
        let one = self.constant(1, 2);
        self.add(x, one)
    }

    /// Boolean AND (Z_2 wires)
    pub fn and(&mut self, x: Wire, y: Wire) -> Wire {
        let nx = self.not(x);
        let left = self.switch(y, nx);
        let zero = self.constant(0, 2);
        let right = self.switch(zero, x);
        self.join(left, right)
    }

    /// Boolean OR (Z_2 wires)
    pub fn or(&mut self, x: Wire, y: Wire) -> Wire {
        let nx = self.not(x);
        let one = self.constant(1, 2);
        let left = self.switch(one, nx);
        let right = self.switch(y, x);
        self.join(left, right)
    }

    // --- Vector operations ---

    /// Element-wise addition of two wire vectors.
    pub fn add_vec(&mut self, x: &[Wire], y: &[Wire]) -> Vec<Wire> {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| self.add(a, b))
            .collect()
    }
}
