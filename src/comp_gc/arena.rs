//! Flat-arena execution of recorded schedules.
//!
//! The worklist engines and tape/journal replayers store per-wire masks and
//! labels as `Vec<Option<Label>>` — an enum behind an `Option` with a heap
//! allocation per k > 1 result. For the streaming pipeline's hot phases
//! (chunk conversion and per-prime extract, all CF wires) that representation
//! is the dominant cost: every gate pays enum dispatch, `Option` bookkeeping
//! and an allocator round-trip around ~30 ns of real lane arithmetic.
//!
//! This module re-executes the same schedules against dense typed arenas:
//!
//! * Z₂ wires live in `Vec<[u64; 2]>` (the packed wire format),
//! * k ∈ [2, 32] wires live in `Vec<[u32; LAMBDA]>` (one lane per coordinate),
//! * a per-shape [`WireLayout`] maps wire id → arena slot once, so execution
//!   never touches `Label` at all; conversions happen only at phase
//!   boundaries (carry items, join diffs, output masks).
//!
//! Arenas persist across phases: a slot is written by its phase's seeds or
//! schedule before any read (the same induction that validates the schedule),
//! so no clearing is needed. A `defined` bitset hard-asserts that invariant —
//! a foreign tape/journal must fail loudly, never read stale state (matching
//! the `garble_replay` / `replay_with_labels` misuse contracts).
//!
//! Security note: the garbler-side and evaluator-side runs use separate
//! arenas (the parties never share state); hashing goes through the same
//! CCRH calls as the `Label` path, with the same solo-domain nonces.

use super::program::Program;
use crate::crypto::expand;
use crate::exec::{JournalEntry, Worklist};
use crate::label::{CfLabel, LAMBDA, Label};
use crate::system::System;
use crate::types::{Gate, Val, Wire};

/// Packed arena location of a wire: bits 31..30 kind (0 = Z₂, 1 = lanes),
/// bits 29..24 the lane width k, bits 23..0 the slot index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot(u32);

const KIND_Z2: u32 = 0;
const KIND_LANES: u32 = 1;

impl Slot {
    fn z2(index: usize) -> Slot {
        debug_assert!(index < (1 << 24));
        Slot((KIND_Z2 << 30) | index as u32)
    }
    fn lanes(index: usize, k: u32) -> Slot {
        debug_assert!(index < (1 << 24) && (2..=32).contains(&k));
        Slot((KIND_LANES << 30) | (k << 24) | index as u32)
    }
    #[inline]
    fn kind(self) -> u32 {
        self.0 >> 30
    }
    #[inline]
    fn k(self) -> u32 {
        (self.0 >> 24) & 0x3f
    }
    #[inline]
    fn index(self) -> usize {
        (self.0 & 0x00ff_ffff) as usize
    }
    #[inline]
    fn is_z2(self) -> bool {
        self.kind() == KIND_Z2
    }
}

/// Per-shape wire → slot map. Built once per phase shape and cached alongside
/// the garble tape; valid for any System with the same wire kinds/moduli.
#[derive(Debug)]
pub(crate) struct WireLayout {
    slots: Vec<Slot>,
    z2_count: usize,
    lanes_count: usize,
}

impl WireLayout {
    /// Build the layout, or `None` if the System uses wires the arena does
    /// not support (NCF wires, moduli > 2^32, or switch groups) — callers
    /// fall back to the `Label` path.
    pub(crate) fn build(system: &System) -> Option<WireLayout> {
        if system.num_switch_groups() > 0 {
            return None;
        }
        let mut slots = Vec::with_capacity(system.num_wires());
        let (mut z2, mut lanes) = (0usize, 0usize);
        for wid in 0..system.num_wires() {
            let w = Wire { wid };
            let m = system.modulus(w);
            if !system.is_cf(w) || !m.is_power_of_two() || m > (1u64 << 32) {
                return None;
            }
            let k = m.trailing_zeros();
            if k == 1 {
                slots.push(Slot::z2(z2));
                z2 += 1;
            } else {
                slots.push(Slot::lanes(lanes, k));
                lanes += 1;
            }
        }
        Some(WireLayout {
            slots,
            z2_count: z2,
            lanes_count: lanes,
        })
    }

    #[inline]
    fn slot(&self, w: Wire) -> Slot {
        self.slots[w.wid]
    }
}

/// Dense mask/label storage for one party. Persistent across phases.
#[derive(Debug, Default)]
pub(crate) struct LabelArena {
    z2: Vec<[u64; 2]>,
    lanes: Vec<[u32; LAMBDA]>,
    /// One bit per wire of the current phase: written this phase.
    defined: Vec<u64>,
    /// CCRH output scratch (zeroed once; `expand` overwrites the used prefix).
    scratch: Vec<u8>,
}

impl LabelArena {
    /// Grow storage (and the hash scratch) to at least the given counts.
    fn ensure(&mut self, z2_count: usize, lanes_count: usize) {
        if self.z2.len() < z2_count {
            self.z2.resize(z2_count, [0; 2]);
        }
        if self.lanes.len() < lanes_count {
            self.lanes.resize(lanes_count, [0; LAMBDA]);
        }
        if self.scratch.len() < 512 {
            self.scratch.resize(512, 0);
        }
    }

    /// Size for `layout`, resetting the per-phase definedness.
    fn begin_phase(&mut self, layout: &WireLayout) {
        if self.z2.len() < layout.z2_count {
            self.z2.resize(layout.z2_count, [0; 2]);
        }
        if self.lanes.len() < layout.lanes_count {
            self.lanes.resize(layout.lanes_count, [0; LAMBDA]);
        }
        let words = layout.slots.len().div_ceil(64);
        self.defined.clear();
        self.defined.resize(words, 0);
        if self.scratch.len() < 512 {
            self.scratch.resize(512, 0);
        }
    }

    #[inline]
    fn mark(&mut self, w: Wire) {
        self.defined[w.wid >> 6] |= 1 << (w.wid & 63);
    }
    #[inline]
    fn is_defined(&self, w: Wire) -> bool {
        (self.defined[w.wid >> 6] >> (w.wid & 63)) & 1 != 0
    }
    /// Hard misuse checks, mirroring the `Label`-path replayers: operands must
    /// be written this phase, the destination must not be.
    fn check_write(&self, dst: Wire, srcs: &[Wire]) {
        assert!(
            !self.is_defined(dst),
            "arena: wire {} already written — tape/journal mismatch",
            dst.wid
        );
        for s in srcs {
            assert!(
                self.is_defined(*s),
                "arena: operand wire {} unwritten — tape/journal mismatch",
                s.wid
            );
        }
    }

    // -- Boundary conversions --

    fn store_label(&mut self, slot: Slot, l: &Label) {
        let c = l.as_cf();
        if slot.is_z2() {
            debug_assert_eq!(c.modulus(), 2);
            let raw = c.raw_bits();
            self.z2[slot.index()] = [raw[0], raw[1]];
        } else {
            debug_assert_eq!(c.modulus(), 1u64 << slot.k());
            self.lanes[slot.index()].copy_from_slice(c.lanes());
        }
    }

    fn load_label(&self, slot: Slot) -> Label {
        if slot.is_z2() {
            let w = self.z2[slot.index()];
            Label::Cf(CfLabel::from_raw_bits(vec![w[0], w[1]], 2))
        } else {
            Label::Cf(CfLabel::from_lanes(
                self.lanes[slot.index()].to_vec(),
                1u64 << slot.k(),
            ))
        }
    }

    // -- Ops (slots must be pre-checked distinct via check_write) --

    #[inline]
    fn xor_z2(&mut self, dst: Slot, a: Slot, b: Slot) {
        let (av, bv) = (self.z2[a.index()], self.z2[b.index()]);
        self.z2[dst.index()] = [av[0] ^ bv[0], av[1] ^ bv[1]];
    }

    /// dst = a ± b over lanes mod 2^k. `sub` selects subtraction.
    ///
    /// `dst` can never alias an operand (it is undefined, operands are
    /// defined), but `a == b` is legal (`add(x, x)`).
    #[inline]
    fn addsub_lanes(&mut self, dst: Slot, a: Slot, b: Slot, sub: bool) {
        let k = dst.k();
        debug_assert!(a.k() == k && b.k() == k);
        let mask = lane_mask(k);
        if a.index() == b.index() {
            let [d, av] = self
                .lanes
                .get_disjoint_mut([dst.index(), a.index()])
                .expect("arena dst must differ from operands");
            if sub {
                d.fill(0); // a - a = 0
            } else {
                for i in 0..LAMBDA {
                    d[i] = av[i].wrapping_add(av[i]) & mask; // a + a
                }
            }
            return;
        }
        let [d, av, bv] = self
            .lanes
            .get_disjoint_mut([dst.index(), a.index(), b.index()])
            .expect("arena dst must differ from operands");
        if sub {
            for i in 0..LAMBDA {
                d[i] = av[i].wrapping_sub(bv[i]) & mask;
            }
        } else {
            for i in 0..LAMBDA {
                d[i] = av[i].wrapping_add(bv[i]) & mask;
            }
        }
    }

    /// dst = s · a over lanes mod 2^k (s pre-reduced caller-side is not
    /// required; the mask truncates).
    #[inline]
    fn mul_lanes(&mut self, dst: Slot, a: Slot, s: u64) {
        let k = dst.k();
        let mask = lane_mask(k);
        let s32 = (s & mask as u64 as u64) as u32;
        let [d, av] = self
            .lanes
            .get_disjoint_mut([dst.index(), a.index()])
            .expect("arena slots must be distinct");
        for i in 0..LAMBDA {
            d[i] = s32.wrapping_mul(av[i]) & mask;
        }
    }

    fn zero(&mut self, dst: Slot) {
        if dst.is_z2() {
            self.z2[dst.index()] = [0; 2];
        } else {
            self.lanes[dst.index()].fill(0);
        }
    }

    fn copy(&mut self, dst: Slot, a: Slot) {
        debug_assert_eq!(dst.kind(), a.kind());
        if dst.is_z2() {
            self.z2[dst.index()] = self.z2[a.index()];
        } else {
            debug_assert_eq!(dst.k(), a.k());
            let [d, av] = self
                .lanes
                .get_disjoint_mut([dst.index(), a.index()])
                .expect("arena slots must be distinct");
            d.copy_from_slice(av);
        }
    }

    /// dst = (a >> shift) & mask(k_out) lane-wise — covers Mod2k
    /// (shift = 0) and Div2k (mask implied by dst's k). Destination may be Z₂
    /// (pack the surviving bit) or lanes.
    fn shift_mask(&mut self, dst: Slot, a: Slot, shift: u32) {
        debug_assert!(!a.is_z2(), "shift_mask sources are lanes");
        if dst.is_z2() {
            let av = &self.lanes[a.index()];
            let mut out = [0u64; 2];
            for (w, chunk) in av.chunks_exact(64).enumerate() {
                let mut acc = 0u64;
                for (i, &l) in chunk.iter().enumerate() {
                    acc |= (((l >> shift) & 1) as u64) << i;
                }
                out[w] = acc;
            }
            self.z2[dst.index()] = out;
        } else {
            let mask = lane_mask(dst.k());
            let [d, av] = self
                .lanes
                .get_disjoint_mut([dst.index(), a.index()])
                .expect("arena slots must be distinct");
            for i in 0..LAMBDA {
                d[i] = (av[i] >> shift) & mask;
            }
        }
    }

    /// dst = H(ctrl, domain) — the solo-domain CCRH of a Z₂ control,
    /// expanded straight into the destination slot (no `Label`, no heap).
    fn hash_solo_into(&mut self, dst: Slot, ctrl: Slot, domain: u64) {
        debug_assert!(ctrl.is_z2(), "switch controls are CF Z_2");
        debug_assert!(domain < (1u64 << 63), "solo domain uses bit 63");
        let cw = self.z2[ctrl.index()];
        let mut seed = [0u8; 16];
        seed[0..8].copy_from_slice(&cw[0].to_le_bytes());
        seed[8..16].copy_from_slice(&cw[1].to_le_bytes());
        if dst.is_z2() {
            let mut out = [0u8; 16];
            expand(seed, domain, &mut out);
            self.z2[dst.index()] = [
                u64::from_le_bytes(out[0..8].try_into().unwrap()),
                u64::from_le_bytes(out[8..16].try_into().unwrap()),
            ];
        } else {
            let k = dst.k() as usize;
            let bytes = (LAMBDA * k).div_ceil(64) * 8;
            expand(seed, domain, &mut self.scratch[..bytes]);
            let d = &mut self.lanes[dst.index()];
            match k {
                // Byte-aligned widths: lanes are the bytes (the packed wire
                // format is LSB-first, so byte i is coordinate i).
                8 => {
                    for (lane, &b) in d.iter_mut().zip(&self.scratch[..LAMBDA]) {
                        *lane = b as u32;
                    }
                }
                16 => {
                    for (i, lane) in d.iter_mut().enumerate() {
                        *lane =
                            u16::from_le_bytes(self.scratch[i * 2..i * 2 + 2].try_into().unwrap())
                                as u32;
                    }
                }
                #[cfg(target_arch = "aarch64")]
                _ if neon_unpack_ok(k) => {
                    // Even k: every 4-lane group starts byte-aligned
                    // (4k ≡ 0 mod 8), so one static byte-gather/shift pattern
                    // serves all 32 groups.
                    unpack_even_k_neon(&self.scratch, k, d);
                }
                _ => unpack_generic(&self.scratch[..bytes], k, d),
            }
        }
    }

    /// dst = a ± H where H is a freshly hashed pad (switch output/data).
    fn hash_addsub(&mut self, dst: Slot, a: Slot, ctrl: Slot, domain: u64, sub: bool) {
        // The pad lands in dst, then dst = a ∓/± pad in place.
        self.hash_solo_into(dst, ctrl, domain);
        if dst.is_z2() {
            let av = self.z2[a.index()];
            let d = &mut self.z2[dst.index()];
            d[0] ^= av[0];
            d[1] ^= av[1];
        } else {
            let mask = lane_mask(dst.k());
            let [d, av] = self
                .lanes
                .get_disjoint_mut([dst.index(), a.index()])
                .expect("arena slots must be distinct");
            if sub {
                // dst = a - H (H currently in d)
                for i in 0..LAMBDA {
                    d[i] = av[i].wrapping_sub(d[i]) & mask;
                }
            } else {
                for i in 0..LAMBDA {
                    d[i] = d[i].wrapping_add(av[i]) & mask;
                }
            }
        }
    }

    /// dst = a ± diff where diff is a `Label` from the program (joins).
    fn diff_addsub(&mut self, dst: Slot, a: Slot, diff: &Label, sub: bool) {
        let c = diff.as_cf();
        if dst.is_z2() {
            let raw = c.raw_bits();
            let av = self.z2[a.index()];
            self.z2[dst.index()] = [av[0] ^ raw[0], av[1] ^ raw[1]];
        } else {
            let mask = lane_mask(dst.k());
            let dv = c.lanes();
            let [d, av] = self
                .lanes
                .get_disjoint_mut([dst.index(), a.index()])
                .expect("arena slots must be distinct");
            if sub {
                for i in 0..LAMBDA {
                    d[i] = av[i].wrapping_sub(dv[i]) & mask;
                }
            } else {
                for i in 0..LAMBDA {
                    d[i] = av[i].wrapping_add(dv[i]) & mask;
                }
            }
        }
    }

    /// diff = a − b as a `Label` (join-diff emission).
    fn diff_label(&self, a: Slot, b: Slot) -> Label {
        if a.is_z2() {
            let (av, bv) = (self.z2[a.index()], self.z2[b.index()]);
            Label::Cf(CfLabel::from_raw_bits(
                vec![av[0] ^ bv[0], av[1] ^ bv[1]],
                2,
            ))
        } else {
            let mask = lane_mask(a.k());
            let (av, bv) = (&self.lanes[a.index()], &self.lanes[b.index()]);
            let lanes: Vec<u32> = av
                .iter()
                .zip(bv)
                .map(|(&x, &y)| x.wrapping_sub(y) & mask)
                .collect();
            Label::Cf(CfLabel::from_lanes(lanes, 1u64 << a.k()))
        }
    }
}

#[inline]
fn lane_mask(k: u32) -> u32 {
    if k >= 32 { u32::MAX } else { (1u32 << k) - 1 }
}

/// Scalar unpack of LAMBDA packed `k`-bit coordinates from little-endian
/// bytes into u32 lanes (branchless two-word windows; cf. label.rs).
fn unpack_generic(bytes: &[u8], k: usize, dst: &mut [u32; LAMBDA]) {
    let mut padded = [0u64; LAMBDA * 32 / 64 + 1];
    for (w, slot) in padded.iter_mut().enumerate().take(bytes.len() / 8) {
        *slot = u64::from_le_bytes(bytes[w * 8..(w + 1) * 8].try_into().unwrap());
    }
    let mask = lane_mask(k as u32) as u64;
    for (i, lane) in dst.iter_mut().enumerate() {
        let bit = i * k;
        let window = (padded[bit >> 6] as u128) | ((padded[(bit >> 6) + 1] as u128) << 64);
        *lane = ((window >> (bit & 63)) as u64 & mask) as u32;
    }
}

/// True iff [`unpack_even_k_neon`] handles width `k`: even (group bases stay
/// byte-aligned) and every lane's `shift + k` fits the 32-bit gather window
/// (this excludes exactly k = 30, whose lane-1 shift of 6 overflows it).
#[cfg(target_arch = "aarch64")]
fn neon_unpack_ok(k: usize) -> bool {
    if !k.is_multiple_of(2) || !(2..=32).contains(&k) {
        return false;
    }
    (0..4).all(|j| ((j * k) & 7) + k <= 32)
}

/// NEON unpack for even `k` (2 ≤ k ≤ 32): per 4-lane group, one unaligned
/// 16-byte load at the group's (byte-aligned) base, a TBL byte-gather of each
/// lane's 4-byte window, per-lane right shifts and a mask.
///
/// Even k makes every group base byte-aligned (`4k ≡ 0 (mod 8)`), so the
/// gather indices and shifts are group-invariant. `scratch` must be at least
/// `(124·k)/8 + 16` bytes (the 512-byte hash scratch always is for k ≤ 32);
/// lane 3's window ends at byte 3k/8 + 4 ≤ 16 within each load.
#[cfg(target_arch = "aarch64")]
fn unpack_even_k_neon(scratch: &[u8], k: usize, dst: &mut [u32; LAMBDA]) {
    use std::arch::aarch64::*;
    debug_assert!(neon_unpack_ok(k));
    assert!(
        scratch.len() >= (124 * k) / 8 + 16,
        "unpack scratch too short"
    );

    // Lane j of a group reads bytes (j·k)/8 .. +4, then shifts by (j·k) % 8.
    let mut idx = [0u8; 16];
    let mut shifts = [0i32; 4];
    for j in 0..4 {
        let bit = j * k;
        for b in 0..4 {
            idx[j * 4 + b] = ((bit >> 3) + b) as u8;
        }
        shifts[j] = -((bit & 7) as i32);
    }
    // SAFETY: NEON is part of the crate's aarch64 baseline (the CCRH core
    // already requires it). All loads are in-bounds: group g loads 16 bytes
    // at (g·4·k)/8 with g ≤ 31, bounded by the assert above.
    unsafe {
        let idx_v = vld1q_u8(idx.as_ptr());
        let shift_v = vld1q_s32(shifts.as_ptr());
        let mask_v = vdupq_n_u32(lane_mask(k as u32));
        for g in 0..(LAMBDA / 4) {
            let base = (g * 4 * k) >> 3;
            let window = vld1q_u8(scratch.as_ptr().add(base));
            let gathered = vreinterpretq_u32_u8(vqtbl1q_u8(window, idx_v));
            let aligned = vshlq_u32(gathered, shift_v);
            let lanes = vandq_u32(aligned, mask_v);
            vst1q_u32(dst.as_mut_ptr().add(g * 4), lanes);
        }
    }
}

/// One compiled garbler instruction. Operand fields are arena indices (the
/// kind is implied by the opcode); `aux` carries the gate id for hash ops
/// (the solo CCRH domain is `nonce_base + aux`), the shift for shift ops, or
/// the scalar for muls. `k` is the lane width where one is needed.
#[derive(Clone, Copy, Debug)]
struct Instr {
    op: u8,
    k: u8,
    dst: u32,
    a: u32,
    b: u32,
    aux: u32,
}

const OP_XOR_Z2: u8 = 0;
const OP_ADD_L: u8 = 1;
const OP_SUB_L: u8 = 2;
const OP_MUL_L: u8 = 3;
const OP_ZERO_Z2: u8 = 4;
const OP_ZERO_L: u8 = 5;
const OP_COPY_Z2: u8 = 6;
const OP_COPY_L: u8 = 7;
const OP_PACKBIT: u8 = 8;
const OP_SHIFTMASK_L: u8 = 9;
const OP_HASH_XOR_Z2: u8 = 10;
const OP_MUL_Z2: u8 = 13;
const OP_HASH_ADD_L: u8 = 11;
const OP_HASH_SUB_L: u8 = 12;

/// A fully compiled garbler program for one phase shape: constants, input
/// slots, the instruction stream, and the emission lists. Executing it never
/// touches the `System` — see [`garble_compiled`].
#[derive(Debug)]
pub(crate) struct CompiledGarble {
    /// Arena sizing (from the shape's [`WireLayout`]).
    z2_count: usize,
    lanes_count: usize,
    /// `(slot, -c mod 2^k)` per constant wire.
    consts: Vec<(Slot, u32)>,
    /// Declared input slots, in input order.
    inputs: Vec<Slot>,
    instrs: Vec<Instr>,
    /// `(gid, a, b)` per join gate, in gate order (diff = mask_a − mask_b).
    joins: Vec<(u32, Slot, Slot)>,
    outputs: Vec<Slot>,
}

/// Translate a recorded garble tape into a [`CompiledGarble`].
///
/// Performs the same structural validation a checked replay would — wire
/// membership per gate (`bad_wid`) and a definedness simulation over the
/// seed-then-tape order — so executing the compiled form can skip per-step
/// checks: any tape/system mismatch fails loudly here, at compile time.
pub(crate) fn compile_garble(
    system: &System,
    layout: &WireLayout,
    tape: &[JournalEntry],
    input_wires: &[Wire],
    output_wires: &[Wire],
) -> CompiledGarble {
    let mut defined = vec![false; system.num_wires()];
    let mut consts = Vec::new();
    for (wid, d) in defined.iter_mut().enumerate() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        let neg_c = if v.v == 0 { 0 } else { v.modulus - v.v };
        consts.push((layout.slot(Wire { wid }), neg_c as u32));
        *d = true;
    }
    let inputs: Vec<Slot> = input_wires
        .iter()
        .map(|&w| {
            defined[w.wid] = true;
            layout.slot(w)
        })
        .collect();

    let mut instrs = Vec::with_capacity(tape.len());
    for &JournalEntry { gid, wid } in tape {
        let gid = gid as usize;
        let wid = wid as usize;
        let w = Wire { wid };
        let dst = layout.slot(w);
        let bad_wid = || -> ! {
            panic!(
                "compile_garble: tape pairs gate {gid} with wire {wid}, which \
                 the gate does not touch — tape/system structure mismatch"
            )
        };
        let check = |defined: &[bool], srcs: &[Wire]| {
            assert!(
                !defined[wid],
                "compile_garble: wire {wid} already masked — tape/system mismatch"
            );
            for s in srcs {
                assert!(
                    defined[s.wid],
                    "compile_garble: operand wire {} unmasked — tape/system mismatch",
                    s.wid
                );
            }
        };
        let sl = |w: Wire| layout.slot(w);
        let ins = match system.gates[gid] {
            Gate::Add { in0, in1, out } | Gate::Sub { in0, in1, out } => {
                let is_sub = matches!(system.gates[gid], Gate::Sub { .. });
                let (a, b, sub) = if wid == out.wid {
                    (in0, in1, is_sub)
                } else if wid == in0.wid {
                    (out, in1, !is_sub)
                } else if wid == in1.wid {
                    if is_sub {
                        (in0, out, true)
                    } else {
                        (out, in0, true)
                    }
                } else {
                    bad_wid()
                };
                check(&defined, &[a, b]);
                if dst.is_z2() {
                    Instr {
                        op: OP_XOR_Z2,
                        k: 1,
                        dst: dst.index() as u32,
                        a: sl(a).index() as u32,
                        b: sl(b).index() as u32,
                        aux: 0,
                    }
                } else {
                    Instr {
                        op: if sub { OP_SUB_L } else { OP_ADD_L },
                        k: dst.k() as u8,
                        dst: dst.index() as u32,
                        a: sl(a).index() as u32,
                        b: sl(b).index() as u32,
                        aux: 0,
                    }
                }
            }
            Gate::Mul { in0, scalar, out } => {
                if wid != out.wid {
                    bad_wid()
                }
                if scalar == 0 {
                    check(&defined, &[]);
                    Instr {
                        op: if dst.is_z2() { OP_ZERO_Z2 } else { OP_ZERO_L },
                        k: 0,
                        dst: dst.index() as u32,
                        a: 0,
                        b: 0,
                        aux: 0,
                    }
                } else {
                    check(&defined, &[in0]);
                    // Scalar values may differ across same-shape phases (the
                    // r_i coefficients vary per prime), so the instruction
                    // stores the gate id and the scalar is read from the
                    // current System at run time.
                    if dst.is_z2() {
                        Instr {
                            op: OP_MUL_Z2,
                            k: 1,
                            dst: dst.index() as u32,
                            a: sl(in0).index() as u32,
                            b: 0,
                            aux: gid as u32,
                        }
                    } else {
                        Instr {
                            op: OP_MUL_L,
                            k: dst.k() as u8,
                            dst: dst.index() as u32,
                            a: sl(in0).index() as u32,
                            b: 0,
                            aux: gid as u32,
                        }
                    }
                }
            }
            Gate::Mod2k { in0, out, .. } => {
                if wid != out.wid {
                    bad_wid()
                }
                check(&defined, &[in0]);
                let src = sl(in0);
                if src.is_z2() {
                    Instr {
                        op: OP_COPY_Z2,
                        k: 1,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: 0,
                        aux: 0,
                    }
                } else if dst.is_z2() {
                    Instr {
                        op: OP_PACKBIT,
                        k: src.k() as u8,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: 0,
                        aux: 0,
                    }
                } else {
                    Instr {
                        op: OP_SHIFTMASK_L,
                        k: dst.k() as u8,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: 0,
                        aux: 0,
                    }
                }
            }
            Gate::Div2k { in0, k, out } => {
                if wid != out.wid {
                    bad_wid()
                }
                check(&defined, &[in0]);
                let src = sl(in0);
                if k == 0 {
                    Instr {
                        op: if dst.is_z2() { OP_COPY_Z2 } else { OP_COPY_L },
                        k: dst.k() as u8,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: 0,
                        aux: 0,
                    }
                } else if dst.is_z2() {
                    Instr {
                        op: OP_PACKBIT,
                        k: src.k() as u8,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: k,
                        aux: 0,
                    }
                } else {
                    Instr {
                        op: OP_SHIFTMASK_L,
                        k: dst.k() as u8,
                        dst: dst.index() as u32,
                        a: src.index() as u32,
                        b: k,
                        aux: 0,
                    }
                }
            }
            Gate::Switch { data, ctrl, out } => {
                if wid != out.wid && wid != data.wid {
                    bad_wid()
                }
                let (a, sub) = if wid == out.wid {
                    (data, false)
                } else {
                    (out, true)
                };
                check(&defined, &[ctrl, a]);
                if dst.is_z2() {
                    Instr {
                        op: OP_HASH_XOR_Z2,
                        k: 1,
                        dst: dst.index() as u32,
                        a: sl(a).index() as u32,
                        b: sl(ctrl).index() as u32,
                        aux: gid as u32,
                    }
                } else {
                    Instr {
                        op: if sub { OP_HASH_SUB_L } else { OP_HASH_ADD_L },
                        k: dst.k() as u8,
                        dst: dst.index() as u32,
                        a: sl(a).index() as u32,
                        b: sl(ctrl).index() as u32,
                        aux: gid as u32,
                    }
                }
            }
            Gate::Join { .. } => panic!(
                "compile_garble: tape entry for Join gate {gid} — joins never \
                 propagate masks"
            ),
            Gate::SameWire { a, b } => {
                let src = if wid == b.wid {
                    a
                } else if wid == a.wid {
                    b
                } else {
                    bad_wid()
                };
                check(&defined, &[src]);
                Instr {
                    op: if dst.is_z2() { OP_COPY_Z2 } else { OP_COPY_L },
                    k: dst.k() as u8,
                    dst: dst.index() as u32,
                    a: sl(src).index() as u32,
                    b: 0,
                    aux: 0,
                }
            }
        };
        defined[wid] = true;
        instrs.push(ins);
    }

    let mut joins = Vec::new();
    for (gid, &g) in system.gates.iter().enumerate() {
        if let Gate::Join { a, b } = g {
            assert!(
                defined[a.wid] && defined[b.wid],
                "compile_garble: join gate {gid}: side mask unresolved"
            );
            joins.push((gid as u32, layout.slot(a), layout.slot(b)));
        }
    }
    let outputs = output_wires
        .iter()
        .map(|&w| {
            assert!(
                defined[w.wid],
                "compile_garble: output wire {}: mask unresolved",
                w.wid
            );
            layout.slot(w)
        })
        .collect();

    CompiledGarble {
        z2_count: layout.z2_count,
        lanes_count: layout.lanes_count,
        consts,
        inputs,
        instrs,
        joins,
        outputs,
    }
}

/// Execute a [`CompiledGarble`]: the whole phase garble — seeds, derivation,
/// emission — with no per-step checks (the program was validated against the
/// shape at compile time; the shape-key cache hard-asserts the System matches
/// before calling this). The `system` is consulted only for `Mul` scalars,
/// which legally vary across same-shape phases.
#[allow(clippy::too_many_arguments)]
pub(crate) fn garble_compiled(
    system: &System,
    prog: &CompiledGarble,
    arena: &mut LabelArena,
    input_masks: &[Label],
    delta: u128,
    nonce_base: u64,
) -> Program {
    assert_eq!(prog.inputs.len(), input_masks.len());
    arena.ensure(prog.z2_count, prog.lanes_count);
    let dw = [delta as u64, (delta >> 64) as u64];
    for &(slot, neg_c) in &prog.consts {
        if slot.is_z2() {
            arena.z2[slot.index()] = if neg_c == 0 { [0; 2] } else { dw };
        } else {
            let d = &mut arena.lanes[slot.index()];
            for (i, lane) in d.iter_mut().enumerate() {
                *lane = if (delta >> i) & 1 == 1 { neg_c } else { 0 };
            }
        }
    }
    for (&slot, m) in prog.inputs.iter().zip(input_masks) {
        arena.store_label(slot, m);
    }

    for ins in &prog.instrs {
        let (d, a, b) = (ins.dst as usize, ins.a as usize, ins.b as usize);
        match ins.op {
            OP_XOR_Z2 => {
                let (av, bv) = (arena.z2[a], arena.z2[b]);
                arena.z2[d] = [av[0] ^ bv[0], av[1] ^ bv[1]];
            }
            OP_ADD_L | OP_SUB_L => {
                let mask = lane_mask(ins.k as u32);
                if a == b {
                    let [dv, av] = arena
                        .lanes
                        .get_disjoint_mut([d, a])
                        .expect("compiled dst must differ from operands");
                    if ins.op == OP_SUB_L {
                        dv.fill(0);
                    } else {
                        for i in 0..LAMBDA {
                            dv[i] = av[i].wrapping_add(av[i]) & mask;
                        }
                    }
                    continue;
                }
                let [dv, av, bv] = arena
                    .lanes
                    .get_disjoint_mut([d, a, b])
                    .expect("compiled dst must differ from operands");
                if ins.op == OP_SUB_L {
                    for i in 0..LAMBDA {
                        dv[i] = av[i].wrapping_sub(bv[i]) & mask;
                    }
                } else {
                    for i in 0..LAMBDA {
                        dv[i] = av[i].wrapping_add(bv[i]) & mask;
                    }
                }
            }
            OP_MUL_L => {
                let mask = lane_mask(ins.k as u32);
                let Gate::Mul { scalar, .. } = system.gates[ins.aux as usize] else {
                    panic!("compiled Mul at gid {} is not a Mul gate", ins.aux)
                };
                let s = (scalar & mask as u64) as u32;
                let [dv, av] = arena
                    .lanes
                    .get_disjoint_mut([d, a])
                    .expect("compiled slots are distinct");
                for i in 0..LAMBDA {
                    dv[i] = s.wrapping_mul(av[i]) & mask;
                }
            }
            OP_MUL_Z2 => {
                let Gate::Mul { scalar, .. } = system.gates[ins.aux as usize] else {
                    panic!("compiled Mul at gid {} is not a Mul gate", ins.aux)
                };
                arena.z2[d] = if scalar & 1 == 1 { arena.z2[a] } else { [0; 2] };
            }
            OP_ZERO_Z2 => arena.z2[d] = [0; 2],
            OP_ZERO_L => arena.lanes[d].fill(0),
            OP_COPY_Z2 => arena.z2[d] = arena.z2[a],
            OP_COPY_L => {
                let [dv, av] = arena
                    .lanes
                    .get_disjoint_mut([d, a])
                    .expect("compiled slots are distinct");
                dv.copy_from_slice(av);
            }
            OP_PACKBIT => {
                let shift = ins.b;
                let av = &arena.lanes[a];
                let mut out = [0u64; 2];
                for (wd, chunk) in av.chunks_exact(64).enumerate() {
                    let mut acc = 0u64;
                    for (i, &l) in chunk.iter().enumerate() {
                        acc |= (((l >> shift) & 1) as u64) << i;
                    }
                    out[wd] = acc;
                }
                arena.z2[d] = out;
            }
            OP_SHIFTMASK_L => {
                let shift = ins.b;
                let mask = lane_mask(ins.k as u32);
                let [dv, av] = arena
                    .lanes
                    .get_disjoint_mut([d, a])
                    .expect("compiled slots are distinct");
                for i in 0..LAMBDA {
                    dv[i] = (av[i] >> shift) & mask;
                }
            }
            OP_HASH_XOR_Z2 => {
                let dst = Slot::z2(d);
                arena.hash_solo_into(dst, Slot::z2(b), nonce_base + ins.aux as u64);
                let av = arena.z2[a];
                let dv = &mut arena.z2[d];
                dv[0] ^= av[0];
                dv[1] ^= av[1];
            }
            OP_HASH_ADD_L | OP_HASH_SUB_L => {
                let dst = Slot::lanes(d, ins.k as u32);
                arena.hash_solo_into(dst, Slot::z2(b), nonce_base + ins.aux as u64);
                let mask = lane_mask(ins.k as u32);
                let [dv, av] = arena
                    .lanes
                    .get_disjoint_mut([d, a])
                    .expect("compiled slots are distinct");
                if ins.op == OP_HASH_SUB_L {
                    for i in 0..LAMBDA {
                        dv[i] = av[i].wrapping_sub(dv[i]) & mask;
                    }
                } else {
                    for i in 0..LAMBDA {
                        dv[i] = dv[i].wrapping_add(av[i]) & mask;
                    }
                }
            }
            other => unreachable!("compiled opcode {other}"),
        }
    }

    let mut program = Program::with_num_gates(0);
    for &(gid, a, b) in &prog.joins {
        program.set_join_diff(gid as usize, arena.diff_label(a, b));
    }
    for &slot in &prog.outputs {
        program.push_output_mask(arena.load_label(slot));
    }
    program
}

/// Seed the garbler's constant masks (`-c·Δ_R`) and declared inputs.
/// Production garbling runs [`garble_compiled`]; this interpreted path is the
/// checked reference the differential tests compare against.
#[allow(dead_code)]
fn seed_garble(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    input_wires: &[Wire],
    input_masks: &[Label],
    delta: u128,
) {
    assert_eq!(input_wires.len(), input_masks.len());
    let dw = [delta as u64, (delta >> 64) as u64];
    for wid in 0..system.num_wires() {
        let v = system.values[wid];
        if !v.defined {
            continue;
        }
        let w = Wire { wid };
        let slot = layout.slot(w);
        let neg_c = if v.v == 0 { 0 } else { v.modulus - v.v };
        if slot.is_z2() {
            // -c mod 2 = c: mask is c·Δ₂ (Δ's bits as the packed Z₂ label).
            arena.z2[slot.index()] = if neg_c == 0 { [0; 2] } else { dw };
        } else {
            let nc = neg_c as u32;
            let d = &mut arena.lanes[slot.index()];
            for (i, lane) in d.iter_mut().enumerate() {
                *lane = if (delta >> i) & 1 == 1 { nc } else { 0 };
            }
        }
        arena.mark(w);
    }
    for (&w, m) in input_wires.iter().zip(input_masks) {
        arena.store_label(layout.slot(w), m);
        arena.mark(w);
    }
}

/// Garble by replaying a recorded tape against the arena. Equivalent to
/// [`super::garbler::garble_replay`] (same schedule-validity contract, same
/// loud-failure misuse net), with masks in flat storage. `nonce_base` offsets
/// every solo switch hash: callers allocate `system.num_gates()` fresh ids
/// per phase so no two phases share a CCRH nonce (paper App. A, Def. 4).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn garble_tape_arena(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    tape: &[JournalEntry],
    input_wires: &[Wire],
    input_masks: &[Label],
    output_wires: &[Wire],
    delta: u128,
    nonce_base: u64,
) -> Program {
    arena.begin_phase(layout);
    seed_garble(system, layout, arena, input_wires, input_masks, delta);

    for &JournalEntry { gid, wid } in tape {
        step(
            system,
            layout,
            arena,
            gid as usize,
            wid as usize,
            Party::Garbler { nonce_base },
        );
    }

    // Emission: join diffs in gate order, then declared output masks.
    let mut program = Program::with_num_gates(system.num_gates());
    for (gid, &g) in system.gates.iter().enumerate() {
        if let Gate::Join { a, b } = g {
            assert!(
                arena.is_defined(a) && arena.is_defined(b),
                "join gate {gid}: side mask unresolved"
            );
            program.set_join_diff(gid, arena.diff_label(layout.slot(a), layout.slot(b)));
        }
    }
    for &w in output_wires {
        assert!(
            arena.is_defined(w),
            "output wire {}: mask unresolved",
            w.wid
        );
        program.push_output_mask(arena.load_label(layout.slot(w)));
    }
    program
}

/// Evaluate a garbled system in a single fused pass: cleartext values and
/// labels derive together through one worklist.
///
/// Replaces the two-stage evaluator (an `Exec` discovery pass recording a
/// journal, then a label replay of that journal): a direction fires when its
/// *value* operands are known, and inductively every value-known wire is also
/// label-known (seeds set both, and label propagation follows the same gate
/// directions — joins via the program diff, switches only when the cleartext
/// control is 0). Fusing pays the worklist machinery once and never
/// materializes the journal.
///
/// CF-only shapes mean every wire's modulus is `2^k` with `k` in the layout
/// slot, so value arithmetic needs no `Val` or modulus loads. Returns the
/// output wires' labels and cleartext values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_eval_arena(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    vals: &mut Vec<u64>,
    input_wires: &[Wire],
    input_labels: &[Label],
    input_values: &[u64],
    program: &Program,
    output_wires: &[Wire],
    nonce_base: u64,
) -> (Vec<Label>, Vec<u64>) {
    assert_eq!(input_wires.len(), input_labels.len());
    assert_eq!(input_wires.len(), input_values.len());
    arena.begin_phase(layout);
    vals.clear();
    vals.resize(system.num_wires(), 0);

    let mut wl = Worklist::new(system.num_wires(), system.num_gates());

    // Constants: label 0, value c.
    for (wid, v) in system.values.iter().enumerate() {
        if v.defined {
            let w = Wire { wid };
            arena.zero(layout.slot(w));
            arena.mark(w);
            vals[wid] = v.v;
            wl.mark_known(wid);
        }
    }
    for ((&w, l), &v) in input_wires.iter().zip(input_labels).zip(input_values) {
        arena.store_label(layout.slot(w), l);
        arena.mark(w);
        vals[w.wid] = v;
        wl.mark_known(w.wid);
    }

    let subs = system.subscriptions();
    for wid in 0..system.num_wires() {
        if wl.wire_known(wid) {
            wl.enqueue_all(subs.of(wid));
        }
    }

    while let Some(gid) = wl.pop_live() {
        if fused_fire(
            system, layout, arena, vals, &mut wl, gid, nonce_base, program,
        ) {
            wl.mark_done(gid);
        }
    }

    let labels = output_wires
        .iter()
        .map(|&w| {
            assert!(arena.is_defined(w), "no label on output wire {}", w.wid);
            arena.load_label(layout.slot(w))
        })
        .collect();
    let values = output_wires.iter().map(|&w| vals[w.wid]).collect();
    (labels, values)
}

/// Value mask for a slot's ring: `2^k − 1`.
#[inline]
fn val_mask(slot: Slot) -> u64 {
    if slot.is_z2() {
        1
    } else {
        (1u64 << slot.k()) - 1
    }
}

/// Fire one gate of the fused pass: for each derivable direction whose value
/// operands are known, set the target's value and label together. Returns
/// true when the gate can never derive a new wire (same done rules as the
/// `Exec`/evaluator worklists).
#[allow(clippy::too_many_arguments)]
fn fused_fire(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    vals: &mut [u64],
    wl: &mut Worklist,
    gid: usize,
    nonce_base: u64,
    program: &Program,
) -> bool {
    // One fused write: value + label + bookkeeping + wakeups.
    macro_rules! set {
        ($w:expr, $v:expr, $label:expr) => {{
            let w = $w;
            debug_assert!(!wl.wire_known(w.wid));
            vals[w.wid] = $v;
            $label;
            arena.mark(w);
            wl.mark_known(w.wid);
            wl.wake_subscribers(system.subscriptions().of(w.wid), gid);
        }};
    }
    let known = |wl: &Worklist, w: Wire| wl.wire_known(w.wid);
    match system.gates[gid] {
        Gate::Add { in0, in1, out } | Gate::Sub { in0, in1, out } => {
            let is_sub = matches!(system.gates[gid], Gate::Sub { .. });
            // Fire every still-underived direction of out = in0 ± in1 whose
            // operands are known (same as Exec; at most one fires per visit,
            // after which the remaining targets are known).
            let dirs: [(Wire, Wire, Wire, bool); 3] = [
                (out, in0, in1, is_sub),  // out = in0 ± in1
                (in0, out, in1, !is_sub), // in0 = out ∓ in1
                if is_sub {
                    (in1, in0, out, true)
                } else {
                    (in1, out, in0, true)
                },
            ];
            for (dst, a, b, sub) in dirs {
                if !known(wl, dst) && known(wl, a) && known(wl, b) {
                    let ds = layout.slot(dst);
                    let m = val_mask(ds);
                    let v = if sub {
                        vals[a.wid].wrapping_sub(vals[b.wid]) & m
                    } else {
                        vals[a.wid].wrapping_add(vals[b.wid]) & m
                    };
                    set!(dst, v, {
                        if ds.is_z2() {
                            arena.xor_z2(ds, layout.slot(a), layout.slot(b));
                        } else {
                            arena.addsub_lanes(ds, layout.slot(a), layout.slot(b), sub);
                        }
                    });
                }
            }
            known(wl, in0) && known(wl, in1) && known(wl, out)
        }
        Gate::Mul { in0, scalar, out } => {
            if !known(wl, out) && known(wl, in0) {
                let ds = layout.slot(out);
                let v = scalar.wrapping_mul(vals[in0.wid]) & val_mask(ds);
                set!(out, v, {
                    let src = layout.slot(in0);
                    if scalar == 0 {
                        arena.zero(ds);
                    } else if ds.is_z2() {
                        if scalar & 1 == 1 {
                            arena.copy(ds, src);
                        } else {
                            arena.zero(ds);
                        }
                    } else {
                        arena.mul_lanes(ds, src, scalar);
                    }
                });
            }
            known(wl, out)
        }
        Gate::Mod2k { in0, out, .. } => {
            if !known(wl, out) && known(wl, in0) {
                let ds = layout.slot(out);
                let v = vals[in0.wid] & val_mask(ds);
                set!(out, v, {
                    let src = layout.slot(in0);
                    if src.is_z2() {
                        arena.copy(ds, src);
                    } else {
                        arena.shift_mask(ds, src, 0);
                    }
                });
            }
            known(wl, out)
        }
        Gate::Div2k { in0, k, out } => {
            if !known(wl, out) && known(wl, in0) {
                let ds = layout.slot(out);
                debug_assert!(
                    k == 0 || vals[in0.wid] & ((1u64 << k) - 1) == 0,
                    "div2k: {} not divisible by 2^{}",
                    vals[in0.wid],
                    k
                );
                let v = (vals[in0.wid] >> k) & val_mask(ds);
                set!(out, v, {
                    let src = layout.slot(in0);
                    if k == 0 {
                        arena.copy(ds, src);
                    } else {
                        arena.shift_mask(ds, src, k);
                    }
                });
            }
            known(wl, out)
        }
        Gate::Switch { data, ctrl, out } => {
            if !known(wl, ctrl) {
                return false;
            }
            if vals[ctrl.wid] != 0 {
                return true; // open forever: values are fixed mid-run
            }
            let domain = nonce_base + gid as u64;
            if !known(wl, out) && known(wl, data) {
                let v = vals[data.wid];
                set!(out, v, {
                    arena.hash_addsub(
                        layout.slot(out),
                        layout.slot(data),
                        layout.slot(ctrl),
                        domain,
                        false,
                    )
                });
            }
            if !known(wl, data) && known(wl, out) {
                let v = vals[out.wid];
                set!(data, v, {
                    arena.hash_addsub(
                        layout.slot(data),
                        layout.slot(out),
                        layout.slot(ctrl),
                        domain,
                        true,
                    )
                });
            }
            known(wl, data) && known(wl, out)
        }
        Gate::Join { a, b } => {
            let (ka, kb) = (known(wl, a), known(wl, b));
            if ka != kb {
                // diff = X_a − X_b ⇒ label_b = label_a − diff, label_a = label_b + diff.
                let diff = program
                    .join_diff(gid)
                    .unwrap_or_else(|| panic!("missing join diff for gate {gid}"));
                if ka {
                    set!(b, vals[a.wid], {
                        arena.diff_addsub(layout.slot(b), layout.slot(a), diff, true)
                    });
                } else {
                    set!(a, vals[b.wid], {
                        arena.diff_addsub(layout.slot(a), layout.slot(b), diff, false)
                    });
                }
            }
            known(wl, a) && known(wl, b)
        }
        Gate::SameWire { a, b } => {
            if !known(wl, b) && known(wl, a) {
                set!(b, vals[a.wid], arena.copy(layout.slot(b), layout.slot(a)));
            }
            if !known(wl, a) && known(wl, b) {
                set!(a, vals[b.wid], arena.copy(layout.slot(a), layout.slot(b)));
            }
            known(wl, a) && known(wl, b)
        }
    }
}

/// Evaluate by replaying the cleartext-execution journal against the arena
/// (production evaluation runs [`fused_eval_arena`]; this is the checked
/// reference the differential tests compare against).
#[allow(dead_code)]
#[doc(hidden)]
/// Original doc:
/// Equivalent to [`super::evaluator::replay_with_labels`] (same journal
/// contract and misuse net), with labels in flat storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_journal_arena(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    journal: &[JournalEntry],
    input_wires: &[Wire],
    input_labels: &[Label],
    values: &[Val],
    program: &Program,
    output_wires: &[Wire],
    nonce_base: u64,
) -> Vec<Label> {
    arena.begin_phase(layout);
    // Constants: the evaluator's label is 0.
    for wid in 0..system.num_wires() {
        if system.values[wid].defined {
            let w = Wire { wid };
            arena.zero(layout.slot(w));
            arena.mark(w);
        }
    }
    for (&w, l) in input_wires.iter().zip(input_labels) {
        arena.store_label(layout.slot(w), l);
        arena.mark(w);
    }

    for &JournalEntry { gid, wid } in journal {
        step(
            system,
            layout,
            arena,
            gid as usize,
            wid as usize,
            Party::Evaluator {
                nonce_base,
                values,
                program,
            },
        );
    }

    output_wires
        .iter()
        .map(|&w| {
            assert!(arena.is_defined(w), "no label on output wire {}", w.wid);
            arena.load_label(layout.slot(w))
        })
        .collect()
}

/// Which side's semantics a `step` executes. The two differ only at switches
/// (the evaluator checks the cleartext control and the garbler never reads
/// values) and joins (mask-side: never in a tape; label-side: via the diff).
enum Party<'a> {
    /// Test-only (production garbling is compiled); see `garble_tape_arena`.
    #[allow(dead_code)]
    Garbler { nonce_base: u64 },
    Evaluator {
        nonce_base: u64,
        values: &'a [Val],
        program: &'a Program,
    },
}

/// Execute one schedule entry: gate `gid` derives wire `wid`.
fn step(
    system: &System,
    layout: &WireLayout,
    arena: &mut LabelArena,
    gid: usize,
    wid: usize,
    party: Party<'_>,
) {
    let w = Wire { wid };
    let dst = layout.slot(w);
    let bad_wid = || -> ! {
        panic!(
            "arena: schedule pairs gate {gid} with wire {wid}, which the gate \
             does not touch — tape/journal mismatch"
        )
    };
    match system.gates[gid] {
        Gate::Add { in0, in1, out } | Gate::Sub { in0, in1, out } => {
            let is_sub = matches!(system.gates[gid], Gate::Sub { .. });
            // Derivation direction relative to out = in0 ± in1.
            let (a, b, sub) = if wid == out.wid {
                (in0, in1, is_sub)
            } else if wid == in0.wid {
                (out, in1, !is_sub)
            } else if wid == in1.wid {
                if is_sub {
                    (in0, out, true) // in1 = in0 - out
                } else {
                    (out, in0, true) // in1 = out - in0
                }
            } else {
                bad_wid()
            };
            arena.check_write(w, &[a, b]);
            if dst.is_z2() {
                arena.xor_z2(dst, layout.slot(a), layout.slot(b));
            } else {
                arena.addsub_lanes(dst, layout.slot(a), layout.slot(b), sub);
            }
        }
        Gate::Mul { in0, scalar, out } => {
            if wid != out.wid {
                bad_wid()
            }
            if scalar == 0 {
                arena.check_write(w, &[]);
                arena.zero(dst);
            } else {
                arena.check_write(w, &[in0]);
                let src = layout.slot(in0);
                if dst.is_z2() {
                    // s odd → copy, s even → zero (s·x mod 2).
                    if scalar & 1 == 1 {
                        arena.copy(dst, src);
                    } else {
                        arena.zero(dst);
                    }
                } else {
                    arena.mul_lanes(dst, src, scalar);
                }
            }
        }
        Gate::Mod2k { in0, out, .. } => {
            if wid != out.wid {
                bad_wid()
            }
            arena.check_write(w, &[in0]);
            let src = layout.slot(in0);
            if src.is_z2() {
                arena.copy(dst, src); // k_in = k_out = 1
            } else {
                arena.shift_mask(dst, src, 0);
            }
        }
        Gate::Div2k { in0, k, out } => {
            if wid != out.wid {
                bad_wid()
            }
            arena.check_write(w, &[in0]);
            let src = layout.slot(in0);
            if k == 0 {
                arena.copy(dst, src);
            } else {
                arena.shift_mask(dst, src, k);
            }
        }
        Gate::Switch { data, ctrl, out } => {
            if wid != out.wid && wid != data.wid {
                bad_wid()
            }
            let domain = match &party {
                Party::Garbler { nonce_base } => nonce_base + gid as u64,
                Party::Evaluator {
                    nonce_base, values, ..
                } => {
                    // Exec fires a switch only when its control value is 0.
                    assert_eq!(values[ctrl.wid].v, 0, "replay: switch {gid} open");
                    nonce_base + gid as u64
                }
            };
            let (a, sub) = if wid == out.wid {
                (data, false) // out = data + H
            } else {
                (out, true) // data = out - H
            };
            arena.check_write(w, &[ctrl, a]);
            arena.hash_addsub(dst, layout.slot(a), layout.slot(ctrl), domain, sub);
        }
        Gate::Join { a, b } => match &party {
            Party::Garbler { .. } => {
                panic!("arena: tape entry for Join gate {gid} — joins never propagate masks")
            }
            Party::Evaluator { program, .. } => {
                // diff = X_a − X_b ⇒ label_b = label_a − diff, label_a = label_b + diff.
                let diff = program
                    .join_diff(gid)
                    .unwrap_or_else(|| panic!("missing join diff for gate {gid}"));
                let (src, sub) = if wid == b.wid {
                    (a, true)
                } else if wid == a.wid {
                    (b, false)
                } else {
                    bad_wid()
                };
                arena.check_write(w, &[src]);
                arena.diff_addsub(dst, layout.slot(src), diff, sub);
            }
        },
        Gate::SameWire { a, b } => {
            let src = if wid == b.wid {
                a
            } else if wid == a.wid {
                b
            } else {
                bad_wid()
            };
            arena.check_write(w, &[src]);
            arena.copy(dst, layout.slot(src));
        }
    }
    arena.mark(w);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp_gc::convert::{compute_sub_widths, fold_to_mod_ohe, sub_chunk_extract};
    use crate::comp_gc::garbler::{garble, garble_recorded};
    use crate::comp_gc::{eval_with_labels, replay_with_labels};
    use crate::exec::Exec;
    use crate::label::{self};
    use crate::pipeline::sample_cf_mask;
    use rand::Rng;

    /// The NEON even-k unpacker must agree with the scalar window loop on
    /// every even width and arbitrary bytes.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_unpack_even_k_neon_matches_generic() {
        let mut rng = rand::rng();
        let mut scratch = vec![0u8; 512];
        for round in 0..8 {
            rng.fill(&mut scratch[..]);
            for k in (2..=32).step_by(2).filter(|&k| neon_unpack_ok(k)) {
                let bytes = (LAMBDA * k).div_ceil(64) * 8;
                let mut want = [0u32; LAMBDA];
                unpack_generic(&scratch[..bytes], k, &mut want);
                let mut got = [0u32; LAMBDA];
                unpack_even_k_neon(&scratch, k, &mut got);
                assert_eq!(want, got, "k={k}, round={round}");
            }
            assert!(!neon_unpack_ok(30) && !neon_unpack_ok(7));
        }
    }

    /// Arena garble + eval must agree bit-for-bit with the `Label`-path
    /// reference on the real header circuitry (circular word_to_hot, Mul(0)
    /// seeding, switches, joins, Z₂ and lanes wires), across primes, inputs
    /// and deltas.
    #[test]
    fn test_arena_matches_label_path() {
        let mut rng = rand::rng();
        let ell: u32 = 8;
        let sub_widths = compute_sub_widths(ell, 4); // [4, 4]: forces a peel
        for p in [2u64, 5, 7] {
            for round in 0..3 {
                let input: u64 = rng.random_range(0..(1u64 << ell));
                let delta: u128 = rng.random();
                let nonce_base = round as u64 * 100_000;

                let mut sys = System::new();
                let bit_wires: Vec<Wire> = (0..ell).map(|_| sys.input(2)).collect();
                let x = crate::comp_gc::convert::bin_to_word(&mut sys, &bit_wires, ell);
                let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);
                let h_p = fold_to_mod_ohe(&mut sys, &extraction, p);

                let input_masks: Vec<Label> = bit_wires
                    .iter()
                    .map(|_| sample_cf_mask(&mut rng, 2))
                    .collect();
                let (program_ref, tape) =
                    garble_recorded(&sys, &bit_wires, &input_masks, &h_p, delta, nonce_base);
                // Reference must equal plain garble.
                let program_direct =
                    garble(&sys, &bit_wires, &input_masks, &h_p, delta, nonce_base);
                assert_eq!(program_ref.output_masks(), program_direct.output_masks());

                let layout = WireLayout::build(&sys).expect("CF-only system");
                let mut arena = LabelArena::default();
                let program_arena = garble_tape_arena(
                    &sys,
                    &layout,
                    &mut arena,
                    &tape,
                    &bit_wires,
                    &input_masks,
                    &h_p,
                    delta,
                    nonce_base,
                );
                assert_eq!(
                    program_ref.output_masks(),
                    program_arena.output_masks(),
                    "garbler arena/label divergence: p={p}, input={input}"
                );
                assert_eq!(program_ref.total_bits(), program_arena.total_bits());

                // Compiled program must agree too (including join diffs).
                let compiled = compile_garble(&sys, &layout, &tape, &bit_wires, &h_p);
                let program_compiled =
                    garble_compiled(&sys, &compiled, &mut arena, &input_masks, delta, nonce_base);
                assert_eq!(
                    program_ref.output_masks(),
                    program_compiled.output_masks(),
                    "compiled garble output divergence: p={p}"
                );
                for gid in 0..sys.num_gates() {
                    assert_eq!(
                        program_ref.join_diff(gid),
                        program_compiled.join_diff(gid),
                        "compiled join diff divergence: p={p}, gid={gid}"
                    );
                }

                // Eval side: journal replay, arena vs Label path.
                let d2 = Label::Cf(label::delta_r(delta, 2));
                let input_labels: Vec<Label> = input_masks
                    .iter()
                    .enumerate()
                    .map(|(j, m)| label::add(m, &label::scalar_mul((input >> j) & 1, &d2)))
                    .collect();
                let mut exec = Exec::new(&sys);
                for (j, &w) in bit_wires.iter().enumerate() {
                    exec.set(w, Val::new((input >> j) & 1, 2));
                }
                exec.run_recorded();

                let via_label = replay_with_labels(
                    &sys,
                    &bit_wires,
                    &input_labels,
                    exec.values(),
                    &program_ref,
                    exec.journal(),
                    &h_p,
                    nonce_base,
                );
                let via_worklist = eval_with_labels(
                    &sys,
                    &bit_wires,
                    &input_labels,
                    exec.values(),
                    &program_ref,
                    &h_p,
                    nonce_base,
                );
                assert_eq!(via_label, via_worklist);
                let via_arena = eval_journal_arena(
                    &sys,
                    &layout,
                    &mut arena,
                    exec.journal(),
                    &bit_wires,
                    &input_labels,
                    exec.values(),
                    &program_ref,
                    &h_p,
                    nonce_base,
                );
                assert_eq!(
                    via_label, via_arena,
                    "eval arena/label divergence: p={p}, input={input}"
                );
            }
        }
    }
}
