//! Streaming garble + eval pipeline.
//!
//! The all-at-once `garble` + `eval` pair holds every wire's mask and label
//! in memory until the whole computation is done. For large workloads (e.g.
//! `build_s_aff` at n=256, S=1280) that's tens of millions of wires alive
//! simultaneously even though almost all are short-lived intermediates.
//!
//! `Pipeline` instead executes the computation as a sequence of *phases*:
//!
//! 1. Allocate a fresh `System` for the phase.
//! 2. Re-bind a small carry-forward set of `(mask, label)` pairs from
//!    previous phases as the new System's input wires.
//! 3. Run the caller's gate-construction closure.
//! 4. Garble + evaluate the sub-System.
//! 5. Extract `(mask, label)` for the wires the caller declares as outputs.
//! 6. Drop the System; only the new carry items survive.
//!
//! Peak memory collapses to the largest single phase plus the carry set.
//! Correctness is unchanged: Δ is global and the invariant
//! `label = mask + value · Δ_R(modulus)` holds across phase boundaries.

use super::evaluator::eval_with_labels;
use super::garbler::{garble, normalize_delta};
use super::label::{self, CfLabel, LAMBDA, Label};
use crate::system::System;
use crate::types::Wire;
use rand::Rng;

/// Identifier for a carry-forward wire in the pipeline.
pub type CarryId = usize;

/// One carry-forward wire's state.
///
/// `mask` is the garbler's value for the wire; `label` is the evaluator's.
/// They satisfy `label = mask + value · Δ_R(modulus)`. When a subsequent
/// phase allocates an input wire backed by this carry, the new System's
/// garbler-side mask vector and evaluator-side label vector are seeded with
/// these.
#[derive(Clone, Debug)]
pub struct CarryItem {
    /// Modulus of the wire's ring.
    pub modulus: u64,
    /// True iff the wire is control-friendly.
    pub is_cf: bool,
    /// Garbler's mask.
    pub mask: Label,
    /// Evaluator's label (= mask + value · Δ_R).
    pub label: Label,
}

/// Per-phase telemetry.
#[derive(Clone, Debug, Default)]
pub struct PhaseStats {
    /// Phase name (caller-supplied).
    pub name: String,
    /// Wires allocated inside this phase's System.
    pub wires: usize,
    /// Gates added by this phase.
    pub gates: usize,
    /// NCF switch groups registered by this phase.
    pub switch_groups: usize,
    /// Bits emitted into the program by this phase.
    pub program_bits: usize,
}

use crate::it_gc::BatchCost;

/// Streaming garble + eval orchestrator.
///
/// Lifecycle: build with [`Pipeline::new`], seed inputs via
/// [`seed_input_cf_value`](Pipeline::seed_input_cf_value), call
/// [`run_phase`](Pipeline::run_phase) repeatedly, and decode final outputs
/// with [`decode_ncf`](Pipeline::decode_ncf).
#[derive(Debug)]
pub struct Pipeline {
    /// Global Δ (low bit forced to 1 for point-and-permute).
    pub delta: u128,
    /// Carry slots; `None` after [`drop_carry`](Pipeline::drop_carry).
    carry: Vec<Option<CarryItem>>,

    // --- Aggregate stats across phases. ---
    /// Sum of program bits emitted across all phases.
    pub total_program_bits: usize,
    /// Per-phase stats in execution order.
    pub phase_stats: Vec<PhaseStats>,
    /// Total wires allocated across all phases.
    pub total_wires: usize,
    /// Total gates added across all phases.
    pub total_gates: usize,
    /// Total NCF switch groups registered across all phases.
    pub total_switch_groups: usize,
    /// Maximum `num_wires` of any single phase.
    pub peak_phase_wires: usize,
    /// Maximum `num_gates` of any single phase.
    pub peak_phase_gates: usize,
    /// CF join width summed across phases, in `lg|G|` units (× λ for bits).
    pub join_complexity_cf: usize,
    /// NCF join width summed across phases, in bits.
    pub join_complexity_ncf: usize,
    /// CF hash count summed across phases.
    pub hash_count_cf: usize,
    /// NCF hash count summed across phases (with bulk-pack rebate).
    pub hash_count_ncf: usize,
    /// Wall time in garble (mask propagation + program emit), summed across phases.
    pub garble_secs: f64,
    /// Wall time in eval (label propagation + decode), summed across phases.
    pub eval_secs: f64,
}

impl Pipeline {
    /// New pipeline with a freshly-sampled, normalized Δ.
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        Self::with_delta(normalize_delta(rng.random()))
    }

    /// New pipeline with an explicit Δ. `delta` must already have low bit 1.
    pub fn with_delta(delta: u128) -> Self {
        assert_eq!(delta & 1, 1, "Δ must have low bit 1 (call normalize_delta)");
        Self {
            delta,
            carry: Vec::new(),
            total_program_bits: 0,
            phase_stats: Vec::new(),
            total_wires: 0,
            total_gates: 0,
            total_switch_groups: 0,
            peak_phase_wires: 0,
            peak_phase_gates: 0,
            join_complexity_cf: 0,
            join_complexity_ncf: 0,
            hash_count_cf: 0,
            hash_count_ncf: 0,
            garble_secs: 0.0,
            eval_secs: 0.0,
        }
    }

    /// Seed a CF input wire with a known value (e.g. an input bit).
    ///
    /// Samples a uniform mask, computes `label = mask + value · Δ_R`, and
    /// registers a carry item.
    pub fn seed_input_cf_value<R: Rng>(
        &mut self,
        rng: &mut R,
        modulus: u64,
        value: u64,
    ) -> CarryId {
        assert!(modulus.is_power_of_two());
        assert!(value < modulus);
        let mask = sample_cf_mask(rng, modulus);
        let d = label::delta_r(self.delta, modulus);
        let contrib = label::scalar_mul(value, &Label::Cf(d));
        let label = label::add(&mask, &contrib);
        self.push_carry(CarryItem {
            modulus,
            is_cf: true,
            mask,
            label,
        })
    }

    /// Borrow a carry item.
    pub fn carry(&self, id: CarryId) -> &CarryItem {
        self.carry[id]
            .as_ref()
            .unwrap_or_else(|| panic!("carry id {} already dropped", id))
    }

    /// Free a carry slot. Once dropped, the id can no longer be used.
    pub fn drop_carry(&mut self, id: CarryId) {
        self.carry[id] = None;
    }

    /// Decode an NCF carry item to its concrete value (`(label − mask).rep`).
    pub fn decode_ncf(&self, id: CarryId) -> u64 {
        let item = self.carry(id);
        assert!(!item.is_cf, "decode_ncf: carry {} is CF", id);
        match label::sub(&item.label, &item.mask) {
            Label::Ncf(n) => n.rep,
            Label::Cf(_) => unreachable!(),
        }
    }

    /// Run one phase of the streaming pipeline.
    ///
    /// `inputs` are carry-forward wires consumed by this phase, in the order
    /// they should be allocated as the sub-System's inputs. `build` is the
    /// caller's gate-construction closure: it gets the fresh System and a
    /// slice of input Wires, and returns the wires whose `(mask, label)`
    /// should become new carry items.
    ///
    /// Returns the new carry ids in `build`'s return-vector order.
    pub fn run_phase<F>(
        &mut self,
        phase_name: impl Into<String>,
        inputs: &[CarryId],
        build: F,
    ) -> Vec<CarryId>
    where
        F: FnOnce(&mut System, &[Wire]) -> Vec<Wire>,
    {
        let mut sys = System::new();

        // Bind inputs as fresh wires of matching kind + modulus.
        let input_wires: Vec<Wire> = inputs
            .iter()
            .map(|&id| {
                let item = self.carry(id);
                sys.alloc_wire_kind(item.modulus, item.is_cf)
            })
            .collect();

        let output_wires = build(&mut sys, &input_wires);

        // Garble.
        let input_masks: Vec<Label> = inputs
            .iter()
            .map(|&id| self.carry(id).mask.clone())
            .collect();
        let t_garble = std::time::Instant::now();
        let program = garble(&sys, &input_wires, &input_masks, &output_wires, self.delta);
        self.garble_secs += t_garble.elapsed().as_secs_f64();

        // Evaluate.
        let input_labels: Vec<Label> = inputs
            .iter()
            .map(|&id| self.carry(id).label.clone())
            .collect();
        let t_eval = std::time::Instant::now();
        let output_labels = eval_with_labels(
            &sys,
            &input_wires,
            &input_labels,
            self.delta,
            &program,
            &output_wires,
        );
        self.eval_secs += t_eval.elapsed().as_secs_f64();

        // Output masks live in the program in declaration order.
        let output_masks: Vec<Label> = program.output_masks().to_vec();
        assert_eq!(output_masks.len(), output_wires.len());
        assert_eq!(output_labels.len(), output_wires.len());

        // Aggregate stats.
        let stats = PhaseStats {
            name: phase_name.into(),
            wires: sys.num_wires(),
            gates: sys.num_gates(),
            switch_groups: sys.num_switch_groups(),
            program_bits: program.total_bits(),
        };
        self.total_wires += stats.wires;
        self.total_gates += stats.gates;
        self.total_switch_groups += stats.switch_groups;
        self.peak_phase_wires = self.peak_phase_wires.max(stats.wires);
        self.peak_phase_gates = self.peak_phase_gates.max(stats.gates);
        self.total_program_bits += stats.program_bits;
        self.join_complexity_cf += sys.join_complexity_cf;
        self.join_complexity_ncf += sys.join_complexity_ncf;
        self.hash_count_cf += sys.hash_count_cf;
        self.hash_count_ncf += sys.hash_count_ncf;
        self.phase_stats.push(stats);

        // Promote outputs to carry items.
        let new_ids: Vec<CarryId> = output_wires
            .iter()
            .zip(output_masks)
            .zip(output_labels)
            .map(|((&w, mask), label)| {
                self.push_carry(CarryItem {
                    modulus: sys.modulus(w),
                    is_cf: sys.is_cf(w),
                    mask,
                    label,
                })
            })
            .collect();

        // `sys` and `program` are dropped here.
        new_ids
    }

    /// Fold one externally-garbled batch into the aggregate counters.
    ///
    /// The System-bypass body kernel garbles and evaluates outside
    /// [`run_phase`](Pipeline::run_phase), so it reports its timing and
    /// [`BatchCost`] here instead. This is the kernel-path analogue of the
    /// telemetry `run_phase` records for System phases.
    pub fn record_kernel_batch(&mut self, garble_secs: f64, eval_secs: f64, cost: BatchCost) {
        self.garble_secs += garble_secs;
        self.eval_secs += eval_secs;
        self.total_program_bits += cost.program_bits;
        self.join_complexity_ncf += cost.join_complexity_ncf;
        self.hash_count_ncf += cost.hash_count_ncf;
    }

    fn push_carry(&mut self, item: CarryItem) -> CarryId {
        let id = self.carry.len();
        self.carry.push(Some(item));
        id
    }
}

/// Sample a uniform CF mask in Z_modulus (modulus must be a power of two).
pub fn sample_cf_mask<R: Rng>(rng: &mut R, modulus: u64) -> Label {
    assert!(modulus.is_power_of_two());
    let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
    Label::Cf(CfLabel::from_coords(&coords, modulus))
}
