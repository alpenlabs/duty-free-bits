use crate::label::LAMBDA;
use crate::types::*;

/// The constraint system: holds wires, gates, and propagation queue.
#[derive(Debug)]
pub struct System {
    pub(crate) gates: Vec<Gate>,
    pub(crate) values: Vec<Val>,
    /// Per-wire subscription lists (gates that reference this wire). Inline
    /// `SmallVec` capacity covers the common case (most wires are referenced
    /// by 2-3 gates) without per-wire heap allocation.
    pub(crate) subscriptions: Vec<smallvec::SmallVec<[GateId; 4]>>,
    /// Per-wire control-friendliness flag (parallel to `values`).
    ///
    /// The paper distinguishes **control-friendly** wires (whose labels are
    /// λ-fold to support hash security when used as switch controls) from
    /// **non-control-friendly** wires (single-element labels). This is a *type*
    /// annotation fixed at wire allocation time, not a property of the modulus:
    /// we routinely have CF Z_{2^k} wires and NCF Z_p wires, but the output of
    /// hot-to-ring style constructions over Z_2 can also legitimately be NCF.
    pub(crate) is_cf_flags: Vec<bool>,
    /// NCF switch groups: members sharing a single control wire whose hash is
    /// derived from one bulk CCRH call and sliced across members. Populated by
    /// [`register_ncf_switch_group`]; read by the garbler/evaluator.
    pub(crate) switch_groups: Vec<SwitchGroup>,
    /// For each gate, the (group_index, member_index) it belongs to (if any).
    /// Sized to `num_gates` lazily on first registration.
    pub(crate) gate_to_group: Vec<Option<(u32, u32)>>,
    /// Total join cost in bits (`lg|G|` summed over all joins; CF + NCF).
    /// Communication = LAMBDA·join_complexity_cf + join_complexity_ncf.
    pub join_complexity: usize,
    /// Join width restricted to control-friendly joins (each bit pays λ).
    pub join_complexity_cf: usize,
    /// Join width restricted to non-control-friendly joins (each bit pays 1).
    pub join_complexity_ncf: usize,
    /// CCRH invocations from CF-payload switches. A CF switch on Z_{2^k} costs
    /// k hashes (one λ-bit call per payload bit).
    pub hash_count_cf: usize,
    /// CCRH invocations from NCF-payload switches. Solo NCF switches cost 1
    /// each; a registered switch group of S members in Z_p costs only
    /// ⌈S·lg|R| / λ⌉ — packed by [`register_ncf_switch_group`].
    pub hash_count_ncf: usize,
}

/// A group of NCF switches that share one control wire.
///
/// The garbler/evaluator derives all members' hashes from a single wide CCRH
/// call keyed on the shared control label and the group id, slicing the
/// output across members.
#[derive(Clone, Debug)]
pub struct SwitchGroup {
    /// Shared control wire (must be CF Z_2). Stored as a wire index.
    pub ctrl: Wire,
    /// Member switch gate ids, in slice order (member i gets bits
    /// `[i·lg|R| .. (i+1)·lg|R|)` of the wide hash output).
    pub members: Vec<GateId>,
    /// Common payload modulus across members (NCF, so any modulus ≥ 2).
    pub modulus: u64,
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
            is_cf_flags: Vec::new(),
            switch_groups: Vec::new(),
            gate_to_group: Vec::new(),
            join_complexity: 0,
            join_complexity_cf: 0,
            join_complexity_ncf: 0,
            hash_count_cf: 0,
            hash_count_ncf: 0,
        }
    }

    /// Allocate a wire with explicit CF/NCF kind.
    ///
    /// CF wires require a power-of-two modulus (they must live in a product of
    /// integer rings Z_{2^k}). NCF wires can have any finite modulus.
    pub fn alloc_wire_kind(&mut self, modulus: u64, is_cf: bool) -> Wire {
        if is_cf {
            assert!(
                modulus.is_power_of_two(),
                "CF wire requires power-of-two modulus (got {})",
                modulus
            );
        }
        let wid = self.values.len();
        self.subscriptions.push(smallvec::SmallVec::new());
        self.values.push(Val::none(modulus));
        self.is_cf_flags.push(is_cf);
        Wire { wid }
    }

    /// Allocate a fresh wire in Z_modulus (initially undefined).
    ///
    /// Defaults: CF iff `modulus` is a power of two. Use [`alloc_wire_kind`] to
    /// override — e.g. to allocate a NCF Z_{2^k} wire.
    pub fn alloc_wire(&mut self, modulus: u64) -> Wire {
        self.alloc_wire_kind(modulus, modulus.is_power_of_two())
    }

    fn subscribe(&mut self, w: Wire, gid: GateId) {
        self.subscriptions[w.wid].push(gid);
    }

    /// Get the modulus for a wire.
    pub fn modulus(&self, x: Wire) -> u64 {
        self.values[x.wid].modulus
    }

    /// True iff the wire was allocated as control-friendly.
    pub fn is_cf(&self, x: Wire) -> bool {
        self.is_cf_flags[x.wid]
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

    /// Register a group of NCF switch gates that share one control wire.
    ///
    /// The garbler/evaluator will derive each member's hash from a single wide
    /// CCRH call keyed on the shared control label and the group id, slicing
    /// the output across members. This rebates the NCF hash count from the
    /// naive `members.len()` to `⌈members.len()·lg|R| / λ⌉`.
    ///
    /// Requirements (asserted): every member is a `Switch` gate, every member
    /// is NCF on the same modulus, every member's control wire is `ctrl`, and
    /// no member is already in another group. Returns the new group id.
    pub fn register_ncf_switch_group(&mut self, ctrl: Wire, members: Vec<GateId>) -> usize {
        assert_eq!(self.modulus(ctrl), 2, "switch group control must be Z_2");
        assert!(self.is_cf(ctrl), "switch group control must be CF");
        assert!(!members.is_empty(), "switch group must be non-empty");

        // Resize gate→group map to match num_gates lazily.
        if self.gate_to_group.len() < self.gates.len() {
            self.gate_to_group.resize(self.gates.len(), None);
        }

        let modulus = {
            let out = match self.gates[members[0]] {
                Gate::Switch { out, .. } => out,
                _ => panic!("switch group member must be a Switch"),
            };
            assert!(!self.is_cf(out), "switch group is NCF-only");
            self.modulus(out)
        };
        for &gid in &members {
            let (gate_ctrl, out) = match self.gates[gid] {
                Gate::Switch { ctrl, out, .. } => (ctrl, out),
                _ => panic!("switch group member must be a Switch"),
            };
            assert!(!self.is_cf(out), "switch group is NCF-only");
            assert_eq!(self.modulus(out), modulus, "group member moduli mismatch");
            assert_eq!(gate_ctrl.wid, ctrl.wid, "group member control mismatch");
            assert!(self.gate_to_group[gid].is_none(), "gate {gid} already grouped");
        }

        let group_idx = self.switch_groups.len();
        let s = members.len();
        for (member_idx, &gid) in members.iter().enumerate() {
            self.gate_to_group[gid] = Some((group_idx as u32, member_idx as u32));
        }

        // Rebate the hash count: replace the s solo NCF charges with the bulk total.
        let lg_m = if modulus <= 1 {
            0
        } else {
            ((modulus - 1).ilog2() + 1) as usize
        };
        let bulk = (s * lg_m).div_ceil(LAMBDA);
        self.hash_count_ncf -= s;
        self.hash_count_ncf += bulk;

        self.switch_groups.push(SwitchGroup { ctrl, members, modulus });
        group_idx
    }

    /// Look up a gate's group membership, if any.
    pub fn gate_group(&self, gid: GateId) -> Option<(usize, usize)> {
        if gid < self.gate_to_group.len() {
            self.gate_to_group[gid].map(|(g, m)| (g as usize, m as usize))
        } else {
            None
        }
    }

    /// Borrow a switch group by index.
    pub fn switch_group(&self, idx: usize) -> &SwitchGroup {
        &self.switch_groups[idx]
    }

    /// Number of registered switch groups.
    pub fn num_switch_groups(&self) -> usize {
        self.switch_groups.len()
    }

    // --- Wire constructors ---

    /// Create a fresh input wire in Z_modulus (default kind: CF iff pow2).
    pub fn input(&mut self, modulus: u64) -> Wire {
        self.alloc_wire(modulus)
    }

    /// NCF input wire in Z_modulus.
    pub fn input_ncf(&mut self, modulus: u64) -> Wire {
        self.alloc_wire_kind(modulus, false)
    }

    /// Convenience: input wire in Z_{2^bl}.
    pub fn input_bits(&mut self, bl: u32) -> Wire {
        self.input(1u64 << bl)
    }

    /// Create a constant wire holding `n` in Z_modulus (default kind: CF iff pow2).
    pub fn constant(&mut self, n: u64, modulus: u64) -> Wire {
        let w = self.alloc_wire(modulus);
        self.values[w.wid] = Val::new(n, modulus);
        w
    }

    /// NCF constant wire holding `n` in Z_modulus.
    pub fn constant_ncf(&mut self, n: u64, modulus: u64) -> Wire {
        let w = self.alloc_wire_kind(modulus, false);
        self.values[w.wid] = Val::new(n, modulus);
        w
    }

    /// Constant wire holding `n`, with modulus and kind inherited from `reference`.
    /// Useful inside composite constructions that need a zero (or other literal)
    /// in the same ring+kind as an external wire — e.g. the `z` in `ohe_scale`.
    pub fn constant_matching(&mut self, n: u64, reference: Wire) -> Wire {
        let m = self.modulus(reference);
        let w = self.alloc_wire_kind(m, self.is_cf(reference));
        self.values[w.wid] = Val::new(n, m);
        w
    }

    /// Convenience: constant wire in Z_{2^bl}.
    pub fn constant_bits(&mut self, n: u64, bl: u32) -> Wire {
        self.constant(n, 1u64 << bl)
    }

    // --- Gate management ---

    fn add_gate(&mut self, g: Gate) {
        let gid = self.gates.len();
        self.gates.push(g);
        // Subscribe the gate to the wires it reads, so it wakes when they change.
        // One-input gates (Mul/Mod2k/Div2k) are forward-only and subscribe only
        // `in0`; the others subscribe every wire they touch.
        match g {
            Gate::Switch { data, ctrl, out } => {
                self.subscribe(data, gid);
                self.subscribe(ctrl, gid);
                self.subscribe(out, gid);
            }
            Gate::Add { in0, in1, out } | Gate::Sub { in0, in1, out } => {
                self.subscribe(in0, gid);
                self.subscribe(in1, gid);
                self.subscribe(out, gid);
            }
            Gate::Join { a, b } | Gate::SameWire { a, b } => {
                self.subscribe(a, gid);
                self.subscribe(b, gid);
            }
            Gate::Mul { in0, .. } | Gate::Mod2k { in0, .. } | Gate::Div2k { in0, .. } => {
                self.subscribe(in0, gid);
            }
        }
    }

    // --- Core gate constructors ---

    /// Switch: data wire x (any ring), control wire s (must be CF Z_2).
    /// Output inherits x's kind and modulus. If s=0, output=x.
    ///
    /// Cost (charged to hash counters): a CF payload on Z_{2^k} adds k hashes
    /// to `hash_count_cf` (one λ-bit CCRH call per payload bit). An NCF payload
    /// adds 1 hash to `hash_count_ncf`; this can later be rebated by registering
    /// a [`SwitchGroup`] that packs many NCF switches under one wide call.
    pub fn switch(&mut self, x: Wire, s: Wire) -> Wire {
        assert_eq!(self.modulus(s), 2, "switch control must be binary (Z_2)");
        assert!(self.is_cf(s), "switch control must be CF");
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        let m = self.modulus(x);
        if self.is_cf(x) {
            self.hash_count_cf += if m <= 1 { 0 } else { m.trailing_zeros() as usize };
        } else {
            self.hash_count_ncf += 1;
        }
        self.add_gate(Gate::Switch { data: x, ctrl: s, out });
        out
    }

    /// Join: constrain x = y. Both must have the same modulus and kind.
    /// Costs `(λ if CF else 1) · lg|G|` bits of join width in the paper; we
    /// track the `lg|G|` factor here and leave the λ multiplier to garble-time.
    ///
    /// `join_complexity_cf` and `join_complexity_ncf` give the split (still in
    /// `lg|G|` units). Total program join bits = LAMBDA·cf + ncf.
    pub fn join(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "join: kind mismatch");
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
        self.add_gate(Gate::Join { a: x, b: y });
        x
    }

    /// SameWire: constrain x = y without join cost (when one side is unconstrained).
    /// Both must share modulus and kind.
    pub fn same_wire(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "same_wire: kind mismatch");
        self.add_gate(Gate::SameWire { a: x, b: y });
        x
    }

    /// Addition in the same ring; both inputs must share kind.
    pub fn add(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "add: kind mismatch");
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Add { in0: x, in1: y, out });
        out
    }

    /// Subtraction in the same ring; both inputs must share kind.
    pub fn sub(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "sub: kind mismatch");
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Sub { in0: x, in1: y, out });
        out
    }

    /// Scalar multiplication by constant s (mod wire's modulus). Output inherits kind.
    pub fn mul(&mut self, s: u64, x: Wire) -> Wire {
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Mul { in0: x, scalar: s, out });
        out
    }

    /// Modular reduction: x mod 2^k. Input must be in Z_{2^n} with k ≤ n. Output inherits kind.
    pub fn mod2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k <= m.ilog2());
        let out = self.alloc_wire_kind(1u64 << k, self.is_cf(x));
        self.add_gate(Gate::Mod2k { in0: x, k, out });
        out
    }

    /// Division by 2^k. Input in Z_{2^{k+c}}, output in Z_{2^c}. Output inherits kind.
    pub fn div2k(&mut self, x: Wire, k: u32) -> Wire {
        let m = self.modulus(x);
        assert!(m.is_power_of_two());
        assert!(k < m.ilog2());
        let out = self.alloc_wire_kind(m >> k, self.is_cf(x));
        self.add_gate(Gate::Div2k { in0: x, k, out });
        out
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
