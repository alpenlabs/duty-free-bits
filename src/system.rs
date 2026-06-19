//! The switch-system constraint graph: wires, gates, and derived operations.
//!
//! A [`System`] is built once per streaming phase (wires allocated, gates
//! appended, constants set) and then garbled/evaluated by the engines in
//! [`crate::comp_gc`]. It also tracks the circuit's communication and hash
//! footprint ([`Cost`]) and per-wire subscription lists (compiled to a CSR on
//! first propagation).

use crate::label::LAMBDA;
use crate::types::*;
use std::sync::OnceLock;

/// The constraint system: holds wires, gates, and propagation queue.
#[derive(Debug)]
pub struct System {
    pub(crate) gates: Vec<Gate>,
    pub(crate) values: Vec<Val>,
    /// Wire→gate subscription edges, as built (one `(wid, gid)` per gate-wire
    /// reference). Read through [`subscriptions`](System::subscriptions),
    /// which compiles them into a CSR on first use.
    sub_edges: Vec<(u32, u32)>,
    /// Lazily-built CSR over `sub_edges` (one flat allocation instead of a
    /// heap `Vec` per wire — the engines walk subscription lists on every
    /// wire update). `OnceLock` keeps `System: Sync`.
    sub_csr: OnceLock<SubscriptionCsr>,
    /// Per-wire control-friendliness flag.
    pub(crate) is_cf_flags: Vec<bool>,
    /// Reduces hash count by grouping switches that share the same control wire.
    pub(crate) switch_groups: Vec<SwitchGroup>,
    /// For each gate, the (group_index, member_index) it belongs to (if any),
    /// reverse-lookup table for switch groups.
    pub(crate) gate_to_group: Vec<Option<(u32, u32)>>,
}

/// Per-wire subscription lists in compressed-sparse-row form: wire `w`'s
/// subscribers are `gids[offsets[w] .. offsets[w + 1]]`.
#[derive(Debug)]
pub(crate) struct SubscriptionCsr {
    offsets: Vec<u32>,
    gids: Vec<u32>,
}

impl SubscriptionCsr {
    fn build(num_wires: usize, edges: &[(u32, u32)]) -> Self {
        assert!(
            num_wires <= u32::MAX as usize && edges.len() <= u32::MAX as usize,
            "subscription CSR uses u32 ids/offsets"
        );
        // Counting sort by wire id; gate order within a wire is preserved
        // (insertion order is preserved).
        let mut offsets = vec![0u32; num_wires + 1];
        for &(wid, _) in edges {
            offsets[wid as usize + 1] += 1;
        }
        for w in 0..num_wires {
            offsets[w + 1] += offsets[w];
        }
        let mut cursor = offsets.clone();
        let mut gids = vec![0u32; edges.len()];
        for &(wid, gid) in edges {
            let c = &mut cursor[wid as usize];
            gids[*c as usize] = gid;
            *c += 1;
        }
        SubscriptionCsr { offsets, gids }
    }

    /// Gates subscribed to wire `w`.
    #[inline]
    pub(crate) fn of(&self, w: usize) -> &[u32] {
        &self.gids[self.offsets[w] as usize..self.offsets[w + 1] as usize]
    }
}

/// Communication + hash cost of a circuit.
/// Switches contribute no communication as the evaluator
/// learns all controls from the cleartext input.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    /// Join width over control-friendly joins, in `lg|G|` units (each bit pays λ).
    pub join_complexity_cf: usize,
    /// Join width over non-control-friendly joins, in bits (each pays 1).
    pub join_complexity_ncf: usize,
    /// CCRH invocations from CF-payload switches (a CF switch on Z_{2^k} costs k).
    pub hash_count_cf: usize,
    /// CCRH invocations from NCF switches: 1 per solo switch, `⌈S·lg|R|/λ⌉` per
    /// `S`-member switch group (bulk-packed).
    pub hash_count_ncf: usize,
}

impl Cost {
    /// Total communication in bits.
    pub fn join_complexity(&self) -> usize {
        self.join_complexity_cf + self.join_complexity_ncf
    }
}

/// A group of NCF switches that share one control wire. TODO: maybe the better abstraction is arbitrary payload in (Z_p1 x ... x Z_pn)
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
            sub_edges: Vec::new(),
            sub_csr: OnceLock::new(),
            is_cf_flags: Vec::new(),
            switch_groups: Vec::new(),
            gate_to_group: Vec::new(),
        }
    }

    /// Clear all construction state, keeping allocations for reuse (the
    /// streaming pipeline builds one System per phase; pooling avoids
    /// re-growing the gate/wire vectors every time).
    pub fn reset(&mut self) {
        self.gates.clear();
        self.values.clear();
        self.sub_edges.clear();
        self.is_cf_flags.clear();
        self.switch_groups.clear();
        self.gate_to_group.clear();
        self.sub_csr = OnceLock::new();
    }

    /// Per-wire subscription lists (gates that reference each wire), compiled
    /// to CSR on first use. Construction must be finished before the first
    /// call (the propagation engines run on a fully-built system).
    pub(crate) fn subscriptions(&self) -> &SubscriptionCsr {
        let csr = self
            .sub_csr
            .get_or_init(|| SubscriptionCsr::build(self.values.len(), &self.sub_edges));
        // The CSR freezes the graph: extending the System after a propagation
        // pass would silently run on stale subscriptions, so fail loudly.
        assert_eq!(
            csr.gids.len(),
            self.sub_edges.len(),
            "System was extended after a propagation pass (gates added after \
             the subscription CSR was built)"
        );
        csr
    }

    /// Compute the circuit's communication + hash cost.
    pub fn cost(&self) -> Cost {
        let mut c = Cost::default();
        for (gid, &g) in self.gates.iter().enumerate() {
            match g {
                Gate::Switch { out, .. } => {
                    if self.is_cf(out) {
                        let m = self.modulus(out);
                        c.hash_count_cf += if m <= 1 {
                            0
                        } else {
                            m.trailing_zeros() as usize
                        };
                    } else if self.gate_group(gid).is_none() {
                        // Solo NCF switch; grouped ones are counted via the group.
                        c.hash_count_ncf += 1;
                    }
                }
                Gate::Join { a, .. } => {
                    let m = self.modulus(a);
                    let bits = if m <= 1 {
                        0
                    } else {
                        (m as u128 - 1).ilog2() as usize + 1
                    };
                    if self.is_cf(a) {
                        c.join_complexity_cf += bits;
                    } else {
                        c.join_complexity_ncf += bits;
                    }
                }
                _ => {}
            }
        }
        for group in &self.switch_groups {
            let lg_m = if group.modulus <= 1 {
                0
            } else {
                (group.modulus - 1).ilog2() as usize + 1
            };
            c.hash_count_ncf += (group.members.len() * lg_m).div_ceil(LAMBDA);
        }
        c
    }

    /// Allocate a wire with explicit CF/NCF kind.
    ///
    /// CF wires require a power-of-two modulus. NCF wires can have any finite modulus.
    pub fn alloc_wire_kind(&mut self, modulus: u64, is_cf: bool) -> Wire {
        if is_cf {
            assert!(
                modulus.is_power_of_two(),
                "CF wire requires power-of-two modulus (got {})",
                modulus
            );
        }
        let wid = self.values.len();
        self.values.push(Val::none(modulus));
        self.is_cf_flags.push(is_cf);
        Wire { wid }
    }

    /// Allocate a fresh wire in Z_modulus (initially undefined).
    ///
    /// Defaults: CF iff `modulus` is a power of two. Use [`alloc_wire_kind`](Self::alloc_wire_kind) to
    /// override — e.g. to allocate a NCF Z_{2^k} wire. TODO: maybe this should be the default?
    pub fn alloc_wire(&mut self, modulus: u64) -> Wire {
        self.alloc_wire_kind(modulus, modulus.is_power_of_two())
    }

    fn subscribe(&mut self, w: Wire, gid: GateId) {
        debug_assert!(
            self.sub_csr.get().is_none(),
            "System was extended after a propagation pass (gate subscribed \
             after the subscription CSR was built)"
        );
        debug_assert!(w.wid <= u32::MAX as usize && gid <= u32::MAX as usize);
        self.sub_edges.push((w.wid as u32, gid as u32));
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
    /// Group indices key bulk-domain CCRH calls and restart at 0 per System;
    /// the streaming kernels draw their bulk ids above
    /// `affine::KERNEL_NONCE_FLOOR` (2^32), so in-System groups own
    /// `[0, 2^32)`. A System registering ≥ 2^32 groups would violate
    /// Definition-4 nonce freshness against the kernels.
    ///
    /// The garbler/evaluator will derive each member's hash from a single wide
    /// CCRH call keyed on the shared control label and the group id, slicing
    /// the output across members. [`System::cost`] charges the group
    /// `⌈members.len()·lg|R| / λ⌉` hashes instead of `members.len()`.
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
            assert!(
                self.gate_to_group[gid].is_none(),
                "gate {gid} already grouped"
            );
        }

        let group_idx = self.switch_groups.len();
        for (member_idx, &gid) in members.iter().enumerate() {
            self.gate_to_group[gid] = Some((group_idx as u32, member_idx as u32));
        }
        self.switch_groups.push(SwitchGroup {
            ctrl,
            members,
            modulus,
        });
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
    /// Switches add no communication (the join width is the only cost; see
    /// [`System::cost`]); a CF-payload switch on Z_{2^k} costs k hashes, an NCF
    /// one costs 1 (rebated when registered in a [`SwitchGroup`]).
    pub fn switch(&mut self, x: Wire, s: Wire) -> Wire {
        assert_eq!(self.modulus(s), 2, "switch control must be binary (Z_2)");
        assert!(self.is_cf(s), "switch control must be CF");
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Switch {
            data: x,
            ctrl: s,
            out,
        });
        out
    }

    /// Join: constrain x = y. Both must have the same modulus and kind.
    /// Costs `(λ if CF else 1) · lg|G|` bits of communication (see [`System::cost`]).
    pub fn join(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "join: kind mismatch");
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
        self.add_gate(Gate::Add {
            in0: x,
            in1: y,
            out,
        });
        out
    }

    /// Subtraction in the same ring; both inputs must share kind.
    pub fn sub(&mut self, x: Wire, y: Wire) -> Wire {
        assert_eq!(self.modulus(x), self.modulus(y));
        assert_eq!(self.is_cf(x), self.is_cf(y), "sub: kind mismatch");
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Sub {
            in0: x,
            in1: y,
            out,
        });
        out
    }

    /// Scalar multiplication by constant s (mod wire's modulus). Output inherits kind.
    pub fn mul(&mut self, s: u64, x: Wire) -> Wire {
        let out = self.alloc_wire_kind(self.modulus(x), self.is_cf(x));
        self.add_gate(Gate::Mul {
            in0: x,
            scalar: s,
            out,
        });
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
