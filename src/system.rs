use crate::types::*;

/// Computational security parameter, in bits. Used by NCF switch packing:
/// a single hash invocation outputs LAMBDA_BITS of pseudorandom material,
/// so an NCF switch with payload P bits costs ⌈P / LAMBDA_BITS⌉ hashes.
pub const LAMBDA_BITS: usize = 128;

/// The constraint system: holds wires, gates, and propagation queue.
#[derive(Debug)]
pub struct System {
    pub(crate) gates: Vec<Gate>,
    pub(crate) values: Vec<Val>,
    pub(crate) subscriptions: Vec<Vec<GateId>>,
    /// Per-wire control-friendly flag. CF wires use bit-decomposed labels
    /// (cost λ per bit of modulus); NCF wires use ring-element labels
    /// (cost ⌈log₂|modulus|⌉ bits per join, no λ factor).
    pub(crate) wire_cf: Vec<bool>,
    /// Total join width in bits (sum of ⌈log₂|modulus|⌉ over all joins).
    pub join_complexity: usize,
    /// Join width restricted to control-friendly joins.
    /// Communication cost in bits = `join_complexity_cf · λ`.
    pub join_complexity_cf: usize,
    /// Join width restricted to non-control-friendly joins.
    /// Communication cost in bits = `join_complexity_ncf` (no λ multiplier).
    pub join_complexity_ncf: usize,
    /// Hash invocations from switches on CF payloads. A CF switch on a
    /// payload in Z_{2^k} costs k hashes (one per payload bit).
    pub hash_count_cf: usize,
    /// Hash invocations from switches on NCF payloads. Counted as 1 per
    /// switch (under the assumption that NCF payloads, all ≪ λ bits in
    /// our setting, are packed into a single ciphertext per switch).
    pub hash_count_ncf: usize,
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
            wire_cf: Vec::new(),
            join_complexity: 0,
            join_complexity_cf: 0,
            join_complexity_ncf: 0,
            hash_count_cf: 0,
            hash_count_ncf: 0,
        }
    }

    /// Allocate a fresh CF wire in Z_modulus (initially undefined).
    pub fn alloc_wire(&mut self, modulus: u64) -> Wire {
        self.alloc_wire_with_cf(modulus, true)
    }

    /// Allocate a fresh wire in Z_modulus with explicit CF flag.
    pub fn alloc_wire_with_cf(&mut self, modulus: u64, cf: bool) -> Wire {
        let wid = self.values.len();
        self.subscriptions.push(Vec::new());
        self.values.push(Val::none(modulus));
        self.wire_cf.push(cf);
        Wire { wid }
    }

    /// Whether this wire is control-friendly.
    pub fn is_cf(&self, w: Wire) -> bool {
        self.wire_cf[w.wid]
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

    /// Create a fresh CF input wire in Z_modulus.
    pub fn input(&mut self, modulus: u64) -> Wire {
        self.alloc_wire(modulus)
    }

    /// Create a fresh NCF input wire in Z_modulus.
    pub fn input_ncf(&mut self, modulus: u64) -> Wire {
        self.alloc_wire_with_cf(modulus, false)
    }

    /// Convenience: CF input wire in Z_{2^bl}
    pub fn input_bits(&mut self, bl: u32) -> Wire {
        self.input(1u64 << bl)
    }

    /// Create a CF constant wire holding `n` in Z_modulus.
    pub fn constant(&mut self, n: u64, modulus: u64) -> Wire {
        self.constant_with_cf(n, modulus, true)
    }

    /// Create an NCF constant wire holding `n` in Z_modulus.
    pub fn constant_ncf(&mut self, n: u64, modulus: u64) -> Wire {
        self.constant_with_cf(n, modulus, false)
    }

    /// Create a constant wire holding `n` in Z_modulus with explicit CF flag.
    pub fn constant_with_cf(&mut self, n: u64, modulus: u64, cf: bool) -> Wire {
        let w = self.alloc_wire_with_cf(modulus, cf);
        self.values[w.wid] = Val::new(n, modulus);
        w
    }

    /// Create a constant `n` whose modulus and CF flag match `like`.
    pub fn constant_like(&mut self, n: u64, like: Wire) -> Wire {
        self.constant_with_cf(n, self.modulus(like), self.is_cf(like))
    }

    /// Convenience: CF constant wire in Z_{2^bl}
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

    fn one_in_one_out_cf(
        &mut self,
        typ: GateType,
        x: Wire,
        param: u64,
        out_mod: u64,
        out_cf: bool,
    ) -> Wire {
        let out = self.alloc_wire_with_cf(out_mod, out_cf);
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

    /// Switch: data wire x (any ring), control wire s (must be CF Z_2).
    /// Output has same modulus and CF flag as x. Hash cost depends on CF:
    ///   - CF payload in Z_{2^k}: k hashes
    ///   - NCF payload: 1 hash (under λ-packing assumption for small payloads)
    pub fn switch(&mut self, x: Wire, s: Wire) -> Wire {
        assert_eq!(self.modulus(s), 2, "switch control must be binary (Z_2)");
        assert!(self.is_cf(s), "switch control must be CF");
        let m = self.modulus(x);
        if self.is_cf(x) {
            let bits = if m <= 1 {
                0
            } else {
                (m as u128 - 1).ilog2() as usize + 1
            };
            self.hash_count_cf += bits;
        } else {
            // NCF payloads in this codebase are all ≪ λ bits, so we pack
            // each switch into a single ciphertext = one hash invocation.
            self.hash_count_ncf += 1;
        }
        let out = self.alloc_wire_with_cf(m, self.is_cf(x));
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

    /// Join: constrain x = y. Both must have the same modulus and CF flag.
    /// CF joins cost ⌈log₂ m⌉ bits of join width (× λ for true comm cost);
    /// NCF joins cost ⌈log₂ m⌉ bits directly.
    pub fn join(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "join: CF/NCF mismatch");
        let m = self.modulus(x);
        let bits = if m <= 1 {
            0
        } else {
            (m as u128 - 1).ilog2() as usize + 1
        };
        self.join_complexity += bits;
        if self.is_cf(x) {
            self.join_complexity_cf += bits;
        } else {
            self.join_complexity_ncf += bits;
        }
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
        assert_eq!(self.is_cf(x), self.is_cf(y), "same_wire: CF/NCF mismatch");
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

    /// Addition in the same ring. Operands must share CF flag; output inherits.
    pub fn add(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "add: CF/NCF mismatch");
        let out = self.alloc_wire_with_cf(self.modulus(x), self.is_cf(x));
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

    /// Subtraction in the same ring. Operands must share CF flag; output inherits.
    pub fn sub(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "sub: CF/NCF mismatch");
        let out = self.alloc_wire_with_cf(self.modulus(x), self.is_cf(x));
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

    /// Scalar multiplication by constant s (mod wire's modulus). Output inherits CF.
    pub fn mul(&mut self, s: u64, x: Wire) -> Wire {
        let cf = self.is_cf(x);
        self.one_in_one_out_cf(GateType::Mul, x, s, self.modulus(x), cf)
    }

    /// Modular reduction: x mod 2^k. Input must be in Z_{2^n} with k ≤ n.
    pub fn mod2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k <= m.ilog2());
        let cf = self.is_cf(x);
        self.one_in_one_out_cf(GateType::Mod2k, x, k as u64, 1u64 << k, cf)
    }

    /// Division by 2^k. Input must be in Z_{2^{k+c}}, output in Z_{2^c}.
    pub fn div2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k < m.ilog2());
        let cf = self.is_cf(x);
        self.one_in_one_out_cf(GateType::Div2k, x, k as u64, m >> k, cf)
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
