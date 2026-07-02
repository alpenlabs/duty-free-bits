//! Integration tests for the switch system, execution engine, and affine maps.
//!
//! Unit tests for individual modules live in their respective `#[cfg(test)]` blocks:
//! - `types.rs`: Val arithmetic
//! - `comp_gc/ohe.rs`: one-hot encoding
//! - `comp_gc/convert.rs`: bin_to_word, sub-chunk extraction, mod-OHE folding
//! - `crt/mod.rs`: CRT parameters and reconstruction
//! - `crt/bigint.rs`: U576 arithmetic

use crate::affine::build_s_aff_streaming;
use crate::crt::bigint::{FIRST_80_PRIMES, U576};
use crate::crt::{CrtParams, GarnerDecoder};
use crate::exec::Exec;
use crate::pipeline::Pipeline;
use crate::system::System;
use crate::types::*;

use rand::{Rng, RngExt};

const SAMPLES: usize = 10;
const MOD: u64 = 16;

fn rng() -> impl Rng {
    rand::rng()
}

/// Assert the streaming pipeline computes `a_j·x + b_j (mod p_i)` for every prime
/// and component, against a direct known-answer oracle (the streaming pipeline is
/// the reference implementation the kernel path is tested against).
fn assert_s_aff_streaming_correct(
    params: &CrtParams,
    a_vals: &[u64],
    b_vals: &[u64],
    x: u64,
    rng: &mut impl Rng,
) {
    let n = params.n as usize;
    let s_dim = a_vals.len();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();
    let input_bits: Vec<u64> = (0..n).map(|j| (x >> j) & 1).collect();

    let mut pipeline = Pipeline::new(rng);
    let bit_ids: Vec<_> = input_bits
        .iter()
        .map(|&b| pipeline.seed_input_cf_value(rng, 2, b))
        .collect();
    let outputs = build_s_aff_streaming(
        &mut pipeline,
        &bit_ids,
        &input_bits,
        params,
        &a_residues,
        &b_residues,
    );

    // Known answer: (a_j·x + b_j) mod p_i, in small modular arithmetic
    // (operands < p_i ≤ 409, so products fit u64).
    for (i, &p_i) in params.primes.iter().enumerate() {
        let x_mod = x % p_i;
        for j in 0..s_dim {
            let expected = ((a_vals[j] % p_i) * x_mod + (b_vals[j] % p_i)) % p_i;
            assert_eq!(
                outputs[i][j], expected,
                "streaming a·x+b mod p mismatch (prime {p_i}, comp {j}, x={x}, S={s_dim})"
            );
        }
    }
}

// ==================== System gate tests (via Exec) ====================

#[test]
fn test_constant_wire() {
    let mut sys = System::new();
    let c = sys.constant(7, 16);
    let exec = Exec::new(&sys);
    assert_eq!(exec.get(c), Val::new(7, 16));
}

#[test]
fn test_constant_bits_wire() {
    let mut sys = System::new();
    let c = sys.constant_bits(3, 4);
    let exec = Exec::new(&sys);
    assert_eq!(exec.get(c), Val::new(3, 16));
}

#[test]
fn test_bitlen() {
    let mut sys = System::new();
    let w = sys.input_bits(5);
    assert_eq!(sys.bitlen(w), 5);
}

#[test]
#[should_panic(expected = "not a power of 2")]
fn test_bitlen_panics_non_power_of_2() {
    let mut sys = System::new();
    let w = sys.input(7);
    sys.bitlen(w);
}

#[test]
fn test_switch_ctrl_zero() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let s = sys.input(2);
        let out = sys.switch(x, s);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.set(s, Val::new(0, 2));
        exec.run();

        assert_eq!(exec.get(out), Val::new(v, MOD));
    }
}

#[test]
fn test_switch_ctrl_one() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let s = sys.input(2);
        let out = sys.switch(x, s);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.set(s, Val::new(1, 2));
        exec.run();

        assert!(exec.get(out).is_none());
    }
}

#[test]
fn test_switch_backward_propagation() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let s = sys.input(2);
        let out = sys.switch(x, s);

        let mut exec = Exec::new(&sys);
        exec.set(s, Val::new(0, 2));
        exec.set(out, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(x), Val::new(v, MOD));
    }
}

#[test]
fn test_add_gate() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rng.random_range(0..MOD);
        let b = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        let out = sys.add(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(a, MOD));
        exec.set(y, Val::new(b, MOD));
        exec.run();

        assert_eq!(exec.get(out), Val::new((a + b) % MOD, MOD));
    }
}

#[test]
fn test_add_backward_propagation() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rng.random_range(0..MOD);
        let b = rng.random_range(0..MOD);
        let sum = (a + b) % MOD;
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        let out = sys.add(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(a, MOD));
        exec.set(out, Val::new(sum, MOD));
        exec.run();

        assert_eq!(exec.get(y), Val::new(b, MOD));
    }
}

#[test]
fn test_sub_gate() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rng.random_range(0..MOD);
        let b = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        let out = sys.sub(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(a, MOD));
        exec.set(y, Val::new(b, MOD));
        exec.run();

        assert_eq!(exec.get(out), Val::new((MOD + a - b) % MOD, MOD));
    }
}

#[test]
fn test_sub_backward_propagation() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rng.random_range(0..MOD);
        let b = rng.random_range(0..MOD);
        let diff = (MOD + a - b) % MOD;
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        let out = sys.sub(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(y, Val::new(b, MOD));
        exec.set(out, Val::new(diff, MOD));
        exec.run();

        assert_eq!(exec.get(x), Val::new(a, MOD));
    }
}

#[test]
fn test_mul_gate() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let s = rng.random_range(0..MOD);
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let out = sys.mul(s, x);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(out), Val::new((s * v) % MOD, MOD));
    }
}

#[test]
fn test_mod2k_gate() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let k = rng.random_range(1..=4);
        let mut sys = System::new();
        let x = sys.input_bits(4);
        let out = sys.mod2k(x, k);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.run();

        let m = 1u64 << k;
        assert_eq!(exec.get(out), Val::new(v % m, m));
    }
}

#[test]
fn test_div2k_gate() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let k = rng.random_range(1..4);
        let d = 1u64 << k;
        let quotient = rng.random_range(0..(MOD / d));
        let v = quotient * d;
        let mut sys = System::new();
        let x = sys.input_bits(4);
        let out = sys.div2k(x, k);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(out), Val::new(quotient, MOD / d));
    }
}

#[test]
fn test_join_propagates() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        sys.join(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(y), Val::new(v, MOD));
    }
}

#[test]
fn test_join_propagates_reverse() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        sys.join(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(y, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(x), Val::new(v, MOD));
    }
}

#[test]
fn test_join_complexity() {
    let mut sys = System::new();
    let x = sys.input(8);
    let y = sys.input(8);
    sys.join(x, y);
    assert_eq!(sys.cost().join_complexity(), 3); // log2(8) = 3
}

#[test]
fn test_cost_fold() {
    let mut sys = System::new();
    let ctrl = sys.input(2);

    // CF switch on Z_16 (k = 4): 4 hashes.
    let cf = sys.input(16);
    sys.switch(cf, ctrl);

    // Two NCF switches on Z_5.
    let g_a = sys.num_gates();
    let na = sys.constant_ncf(0, 5);
    let a = sys.switch(na, ctrl);
    let g_b = sys.num_gates();
    let nb = sys.constant_ncf(0, 5);
    let b = sys.switch(nb, ctrl);

    // CF join on Z_8 (3 bits) and NCF join on Z_5 (3 bits).
    let (x, y) = (sys.input(8), sys.input(8));
    sys.join(x, y);
    sys.join(a, b);

    // Solo: the two NCF switches cost 1 each.
    let c = sys.cost();
    assert_eq!(c.hash_count_cf, 4);
    assert_eq!(c.hash_count_ncf, 2);
    assert_eq!(c.join_complexity_cf, 3);
    assert_eq!(c.join_complexity_ncf, 3);

    // Grouping the two NCF switches packs them: ⌈2·3 / 128⌉ = 1 hash, no double-count.
    sys.register_ncf_switch_group(ctrl, vec![g_a, g_b]);
    assert_eq!(sys.cost().hash_count_ncf, 1);
}

#[test]
fn test_same_wire_propagates() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let v = rng.random_range(0..MOD);
        let mut sys = System::new();
        let x = sys.input(MOD);
        let y = sys.input(MOD);
        sys.same_wire(x, y);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(v, MOD));
        exec.run();

        assert_eq!(exec.get(y), Val::new(v, MOD));
    }
}

// ==================== Boolean ops ====================

#[test]
fn test_not_truth_table() {
    for b in 0..2u64 {
        let mut sys = System::new();
        let x = sys.input(2);
        let out = sys.not(x);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(b, 2));
        exec.run();

        assert_eq!(exec.get(out), Val::new(1 - b, 2), "NOT({b})");
    }
}

#[test]
fn test_and_truth_table() {
    for a in 0..2u64 {
        for b in 0..2u64 {
            let mut sys = System::new();
            let x = sys.input(2);
            let y = sys.input(2);
            let out = sys.and(x, y);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::new(a, 2));
            exec.set(y, Val::new(b, 2));
            exec.run();

            assert_eq!(exec.get(out), Val::new(a & b, 2), "AND({a}, {b})");
        }
    }
}

#[test]
fn test_or_truth_table() {
    for a in 0..2u64 {
        for b in 0..2u64 {
            let mut sys = System::new();
            let x = sys.input(2);
            let y = sys.input(2);
            let out = sys.or(x, y);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::new(a, 2));
            exec.set(y, Val::new(b, 2));
            exec.run();

            assert_eq!(exec.get(out), Val::new(a | b, 2), "OR({a}, {b})");
        }
    }
}

// ==================== Vector ops ====================

#[test]
fn test_add_vec() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a0 = rng.random_range(0..MOD);
        let a1 = rng.random_range(0..MOD);
        let b0 = rng.random_range(0..MOD);
        let b1 = rng.random_range(0..MOD);

        let mut sys = System::new();
        let x0 = sys.input(MOD);
        let x1 = sys.input(MOD);
        let y0 = sys.input(MOD);
        let y1 = sys.input(MOD);
        let out = sys.add_vec(&[x0, x1], &[y0, y1]);

        let mut exec = Exec::new(&sys);
        exec.set(x0, Val::new(a0, MOD));
        exec.set(x1, Val::new(a1, MOD));
        exec.set(y0, Val::new(b0, MOD));
        exec.set(y1, Val::new(b1, MOD));
        exec.run();

        assert_eq!(exec.get(out[0]), Val::new((a0 + b0) % MOD, MOD));
        assert_eq!(exec.get(out[1]), Val::new((a1 + b1) % MOD, MOD));
    }
}

// ==================== System extension after propagation ====================

#[test]
#[should_panic(expected = "extended after a propagation pass")]
fn test_system_extension_after_run_panics() {
    // The subscription CSR freezes the gate graph at the first propagation.
    // Extending the System afterwards must fail loudly (release included) —
    // silently running on stale subscriptions would skip the new gates.
    let mut sys = System::new();
    let x = sys.input(4);
    let y = sys.input(4);
    let _ = sys.add(x, y);
    Exec::new(&sys).run(); // builds the CSR

    let z = sys.input(4);
    let _ = sys.add(x, z); // stale CSR: must be detected...
    let mut exec = Exec::new(&sys);
    exec.set(x, Val::new(1, 4));
    exec.run(); // ...at the next propagation
}

// ==================== Exec isolation ====================

#[test]
fn exec_does_not_mutate_system() {
    let mut sys = System::new();
    let x = sys.input(8);
    let y = sys.input(8);
    sys.add(x, y);

    let mut exec1 = Exec::new(&sys);
    exec1.set(x, Val::new(3, 8));
    exec1.set(y, Val::new(5, 8));
    exec1.run();

    let mut exec2 = Exec::new(&sys);
    exec2.set(x, Val::new(1, 8));
    exec2.set(y, Val::new(2, 8));
    exec2.run();

    assert!(sys.values[x.wid].is_none());
    assert!(sys.values[y.wid].is_none());
}

// ==================== S_{aff-Z_M} integration tests ====================

#[test]
#[ignore]
fn test_s_aff_scaling() {
    // Run with: N=12 S=4 /usr/bin/time -l cargo test --release \
    //   test_s_aff_scaling -- --ignored --nocapture
    //
    // Env vars:
    //   N        — input bit-length (default 8)
    //   S        — number of affine maps (default 1)
    //
    // Drives the streaming pipeline (Pipeline + build_s_aff_streaming), which
    // builds + garbles + evals one phase at a time, dropping intermediate state
    // at every phase boundary.
    use std::time::Instant;

    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .expect("N must be a u32");
    let s_dim: usize = std::env::var("S")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .expect("S must be a usize");

    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    eprintln!(
        "n={n}, S={s_dim}, ell={}, chunk_size={}, num_chunks={}",
        params.ell, params.chunk_size, params.num_chunks
    );

    let mut rng = rng();
    let max_x = 1u64 << n;
    let a_vals: Vec<u64> = (0..s_dim)
        .map(|_| rng.random_range(0..1u64 << 48))
        .collect();
    let b_vals: Vec<u64> = (0..s_dim)
        .map(|_| rng.random_range(0..1u64 << 48))
        .collect();
    let x: u64 = rng.random_range(0..max_x);

    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();

    {
        eprintln!("---- streaming pipeline ----");
        let t_stream = Instant::now();
        let mut pipeline = Pipeline::new(&mut rng);
        let x_bits: Vec<u64> = (0..n).map(|j| (x >> j) & 1).collect();
        let bit_ids: Vec<_> = x_bits
            .iter()
            .map(|&b| pipeline.seed_input_cf_value(&mut rng, 2, b))
            .collect();
        let outputs = build_s_aff_streaming(
            &mut pipeline,
            &bit_ids,
            &x_bits,
            &params,
            &a_residues,
            &b_residues,
        );
        let stream_secs = t_stream.elapsed().as_secs_f64();

        eprintln!(
            "[stream] {:>7.2}s  | {} phases, peak {} wires / {} gates per phase",
            stream_secs,
            pipeline.phase_stats.len(),
            pipeline.peak_phase_wires,
            pipeline.peak_phase_gates,
        );
        eprintln!(
            "          garble: {:.2}s ({:.0}%) | eval: {:.2}s ({:.0}%) | other: {:.2}s",
            pipeline.garble_secs,
            100.0 * pipeline.garble_secs / stream_secs,
            pipeline.eval_secs,
            100.0 * pipeline.eval_secs / stream_secs,
            stream_secs - pipeline.garble_secs - pipeline.eval_secs,
        );
        eprintln!(
            "          totals: {} wires, {} gates, {} switch groups",
            pipeline.total_wires, pipeline.total_gates, pipeline.total_switch_groups,
        );
        // Real garbler->evaluator communication = the join width (switches reveal
        // nothing). CF joins pay λ per lg|G| bit, NCF joins pay 1.
        let comm_bits =
            crate::label::LAMBDA * pipeline.join_complexity_cf + pipeline.join_complexity_ncf;
        eprintln!(
            "          communication (join width): {} bits ({:.2} MB)  [cf {}·λ + ncf {}]",
            comm_bits,
            comm_bits as f64 / 8.0 / 1024.0 / 1024.0,
            pipeline.join_complexity_cf,
            pipeline.join_complexity_ncf,
        );
        // Internal garbled material handled across all phases (join diffs + the
        // per-phase carry masks, which are NOT sent — they are the garbler's state).
        eprintln!(
            "          internal material: {} bits ({:.2} MB)   hash: cf {}, ncf {}",
            pipeline.total_program_bits,
            pipeline.total_program_bits as f64 / 8.0 / 1024.0 / 1024.0,
            pipeline.hash_count_cf,
            pipeline.hash_count_ncf,
        );

        // ---- Per-phase-class timing breakdown ----
        struct Class {
            n: usize,
            build: f64,
            garble: f64,
            exec: f64,
            label_eval: f64,
            wires: usize,
            gates: usize,
        }
        let mut classes: std::collections::BTreeMap<&str, Class> = Default::default();
        for ps in &pipeline.phase_stats {
            let class = if ps.name.starts_with("chunk[") {
                "chunk conversion"
            } else if ps.name.ends_with("/extract") {
                "prime extract"
            } else {
                "other"
            };
            let c = classes.entry(class).or_insert(Class {
                n: 0,
                build: 0.0,
                garble: 0.0,
                exec: 0.0,
                label_eval: 0.0,
                wires: 0,
                gates: 0,
            });
            c.n += 1;
            c.build += ps.build_secs;
            c.garble += ps.garble_secs;
            c.exec += ps.exec_secs;
            c.label_eval += ps.label_eval_secs;
            c.wires += ps.wires;
            c.gates += ps.gates;
        }
        eprintln!("          ---- phase classes ----");
        eprintln!(
            "          {:<18} {:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "class", "n", "build", "garble", "exec", "lbl-eval", "wires", "gates"
        );
        for (name, c) in &classes {
            eprintln!(
                "          {:<18} {:>4} {:>7.3}s {:>7.3}s {:>7.3}s {:>7.3}s {:>10} {:>10}",
                name, c.n, c.build, c.garble, c.exec, c.label_eval, c.wires, c.gates
            );
        }
        eprintln!(
            "          {:<18} {:>4} {:>8} {:>7.3}s {:>8} {:>7.3}s",
            "fold kernel", "-", "-", pipeline.fold_garble_secs, "-", pipeline.fold_eval_secs,
        );
        eprintln!(
            "          {:<18} {:>4} {:>8} {:>7.3}s {:>8} {:>7.3}s",
            "it-gc body kernel",
            "-",
            "-",
            pipeline.kernel_garble_secs,
            "-",
            pipeline.kernel_eval_secs,
        );

        // Verify outputs reconstruct correctly (decoder precomputed once for
        // the prime set, as the production decode path would).
        let t_decode = std::time::Instant::now();
        let decoder = GarnerDecoder::new(&params.primes);
        for s in 0..s_dim {
            let residues: Vec<u64> = outputs.iter().map(|prime_outs| prime_outs[s]).collect();
            let reconstructed = decoder.reconstruct(&residues);
            let expected = a_vals[s] * x + b_vals[s];
            assert_eq!(reconstructed, U576::from_u64(expected));
        }
        eprintln!(
            "ok: streaming reconstructed all {s_dim} affine maps (decode {:.3}s)",
            t_decode.elapsed().as_secs_f64()
        );
    }
}

#[test]
#[ignore]
fn bench_header_decomposition() {
    // Decompose one prime-header phase into its two sub-circuits and time each:
    //   A: r_i accumulation + sub_chunk_extract   (k=8/22 CF label arithmetic)
    //   B: fold_to_mod_ohe                        (p·(ell−8) Z_2 gates + hashes)
    //
    // Run with: cargo test --release bench_header_decomposition -- --ignored --nocapture
    use crate::comp_gc::convert::{
        SubChunkExtraction, bin_to_word, compute_sub_widths, fold_to_mod_ohe, sub_chunk_extract,
    };
    use crate::crt::pow2_mod;

    let n: u32 = 256;
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let ell = params.ell;
    let chunk_size = params.chunk_size as usize;
    let work_mod = 1u64 << ell;
    let sub_widths = compute_sub_widths(ell, 8);

    let mut rng = rng();
    let x_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();

    let mut pipeline = Pipeline::new(&mut rng);
    let bit_ids: Vec<_> = x_bits
        .iter()
        .map(|&b| pipeline.seed_input_cf_value(&mut rng, 2, b))
        .collect();

    // Chunk conversion (as in production).
    let mut chunk_word_ids = Vec::new();
    for c in 0..params.num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(n as usize);
        let ids = &bit_ids[start..end];
        let outs = pipeline.run_phase(format!("chunk[{c}]"), ids, move |sys, ws| {
            let mut bits = ws.to_vec();
            if bits.len() < chunk_size {
                let zero = sys.constant(0, 2);
                while bits.len() < chunk_size {
                    bits.push(zero);
                }
            }
            vec![bin_to_word(sys, &bits, ell)]
        });
        chunk_word_ids.push(outs[0]);
    }

    // For a few representative primes, run extract and fold as separate phases.
    for &p_i in &[2u64, 101, 257, 409] {
        let sw = sub_widths.clone();
        // Phase A: r_i + sub_chunk_extract. Outputs: flattened bits then first_bin_hot.
        let a_ids = pipeline.run_phase(
            format!("p{}/extract", p_i),
            &chunk_word_ids,
            move |sys, chunk_wires| {
                let mut r_i = sys.constant(0, work_mod);
                for (c, &w_c) in chunk_wires.iter().enumerate() {
                    let coeff = pow2_mod((c * chunk_size) as u32, p_i);
                    if coeff > 0 {
                        let term = sys.mul(coeff, w_c);
                        r_i = sys.add(r_i, term);
                    }
                }
                let ex = sub_chunk_extract(sys, r_i, &sw);
                let mut outs: Vec<Wire> = ex.bits.iter().flatten().copied().collect();
                outs.extend_from_slice(&ex.first_bin_hot);
                outs
            },
        );
        let sw = sub_widths.clone();
        let n_bits: usize = sw.iter().map(|&w| w as usize).sum();
        // Phase B: fold. Rebuild the extraction struct from carried wires.
        let _h_p = pipeline.run_phase(format!("p{}/fold", p_i), &a_ids, move |sys, ws| {
            let (bit_ws, hot_ws) = ws.split_at(n_bits);
            let mut bits = Vec::new();
            let mut off = 0;
            for &w in &sw {
                bits.push(bit_ws[off..off + w as usize].to_vec());
                off += w as usize;
            }
            let ex = SubChunkExtraction {
                bits,
                first_bin_hot: hot_ws.to_vec(),
                sub_widths: sw.clone(),
            };
            fold_to_mod_ohe(sys, &ex, p_i)
        });
    }

    eprintln!(
        "{:<14} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "phase", "build", "garble", "exec", "lbl-eval", "wires", "gates"
    );
    for ps in &pipeline.phase_stats {
        if ps.name.starts_with("chunk") {
            continue;
        }
        eprintln!(
            "{:<14} {:>7.4}s {:>7.4}s {:>7.4}s {:>7.4}s {:>9} {:>9}",
            ps.name,
            ps.build_secs,
            ps.garble_secs,
            ps.exec_secs,
            ps.label_eval_secs,
            ps.wires,
            ps.gates
        );
    }
}

#[test]
#[ignore]
fn bench_primitives() {
    // Microbenchmark the primitive ops the garble/eval inner loops are made of.
    // Run with: cargo test --release bench_primitives -- --ignored --nocapture
    use crate::crypto::expand;
    use crate::hash::{extract_ncf, hash_bulk, hash_solo};
    use crate::label::{self, CfLabel, LAMBDA, Label};
    use std::hint::black_box;
    use std::time::Instant;

    let mut rng = rng();
    let mut rand_cf = |modulus: u64| {
        let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
        Label::Cf(CfLabel::from_coords(&coords, modulus))
    };

    fn bench(name: &str, iters: usize, mut f: impl FnMut()) {
        // Warmup.
        for _ in 0..iters / 10 + 1 {
            f();
        }
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        eprintln!("{name:<42} {ns:>10.1} ns/op");
    }

    // CF label arithmetic at the moduli the header actually uses.
    for k in [1u32, 8, 14, 22] {
        let m = 1u64 << k;
        let a = rand_cf(m);
        let b = rand_cf(m);
        bench(&format!("label::add        CF k={k}"), 200_000, || {
            black_box(label::add(black_box(&a), black_box(&b)));
        });
        bench(
            &format!("label::scalar_mul CF k={k} (s=3)"),
            200_000,
            || {
                black_box(label::scalar_mul(black_box(3), black_box(&a)));
            },
        );
        let big = rand_cf(1u64 << 22);
        if k < 22 {
            bench(&format!("label::mod2k      22->k={k}"), 200_000, || {
                black_box(label::mod2k(black_box(&big), k));
            });
        }
    }
    // Clone (the propagation engine clones labels on every gate visit).
    let a22 = rand_cf(1u64 << 22);
    let a1 = rand_cf(2);
    bench("Label::clone      CF k=22", 200_000, || {
        black_box(black_box(&a22).clone());
    });
    bench("Label::clone      CF k=1", 200_000, || {
        black_box(black_box(&a1).clone());
    });

    // CCRH paths.
    let ctrl = rand_cf(2);
    bench("hash_solo -> CF k=1   (1 AES block)", 200_000, || {
        black_box(hash_solo(black_box(&ctrl), 12345, true, 2));
    });
    bench("hash_solo -> CF k=8   (8 AES blocks)", 100_000, || {
        black_box(hash_solo(black_box(&ctrl), 12345, true, 1 << 8));
    });
    bench("hash_solo -> CF k=22  (22 AES blocks)", 50_000, || {
        black_box(hash_solo(black_box(&ctrl), 12345, true, 1 << 22));
    });
    bench("hash_solo -> NCF p=409", 200_000, || {
        black_box(hash_solo(black_box(&ctrl), 12345, false, 409));
    });
    let mut buf16 = [0u8; 16];
    bench("expand 16B (raw AES block)", 500_000, || {
        expand(black_box([7u8; 16]), black_box(99), &mut buf16);
        black_box(&buf16);
    });
    let mut buf144 = [0u8; 144];
    bench("expand 144B (9 blocks: bulk 128x9bit)", 200_000, || {
        expand(black_box([7u8; 16]), black_box(99), &mut buf144);
        black_box(&buf144);
    });
    bench("hash_bulk 128 members p=409 (1152 bits)", 200_000, || {
        black_box(hash_bulk(black_box(&ctrl), 7, 128 * 9));
    });

    // IT-GC kernel inner ops.
    let wide = hash_bulk(&ctrl, 7, 128 * 9);
    bench("extract_ncf p=409 (9-bit slice)", 1_000_000, || {
        black_box(extract_ncf(black_box(&wide), black_box(64), 409));
    });
    bench("u64 mod_mul (a*b)%p", 1_000_000, || {
        let a = black_box(123u64);
        let b = black_box(371u64);
        black_box((a * b) % black_box(409u64));
    });

    // Allocation: what one Vec<u64> label allocation costs.
    bench("alloc Vec<u64> 2 words (k=1 label)", 500_000, || {
        black_box(vec![0u64; 2]);
    });
    bench("alloc Vec<u64> 44 words (k=22 label)", 500_000, || {
        black_box(vec![0u64; 44]);
    });
}

#[test]
#[ignore]
fn bench_stream_loop() {
    // Repeats the reference-path stream so a sampling profiler has a long window.
    // Run: N=256 S=1536 REPS=30 cargo test --release bench_stream_loop -- --ignored --nocapture
    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "256".into())
        .parse()
        .unwrap();
    let s_dim: usize = std::env::var("S")
        .unwrap_or_else(|_| "1536".into())
        .parse()
        .unwrap();
    let reps: usize = std::env::var("REPS")
        .unwrap_or_else(|_| "30".into())
        .parse()
        .unwrap();
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let mut rng = rng();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let x: u64 = rng.random_range(0..u64::MAX >> (64 - n.min(63)));
        let x_bits: Vec<u64> = (0..n).map(|j| (x >> j) & 1).collect();
        let mut pipeline = Pipeline::new(&mut rng);
        let bit_ids: Vec<_> = x_bits
            .iter()
            .map(|&b| pipeline.seed_input_cf_value(&mut rng, 2, b))
            .collect();
        let outputs = build_s_aff_streaming(
            &mut pipeline,
            &bit_ids,
            &x_bits,
            &params,
            &a_residues,
            &b_residues,
        );
        std::hint::black_box(&outputs);
    }
    eprintln!(
        "{} reps, {:.4}s/rep",
        reps,
        t.elapsed().as_secs_f64() / reps as f64
    );
}

#[test]
fn test_streaming_sweep() {
    // Sweep S around the RESIDUE_BATCH_SIZE=128 body-batch boundary (the path the
    // nonce advance touches) and several x, asserting the streaming pipeline
    // computes a·x+b mod p_i correctly each time.
    let primes: [u64; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let params = CrtParams::from_primes(&primes, 16);
    let n = params.n as usize;
    let mut rng = rng();
    let m: u128 = params.primes.iter().map(|&p| p as u128).product();

    for s_dim in [1usize, 127, 128, 129, 256] {
        for _ in 0..2 {
            let a_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..m as u64)).collect();
            let b_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..m as u64)).collect();
            let x: u64 = rng.random_range(0..(1u64 << n));
            assert_s_aff_streaming_correct(&params, &a_vals, &b_vals, x, &mut rng);
        }
    }
}

#[test]
fn test_streaming_edge_regimes() {
    // Parameter regimes outside the main sweep, each end-to-end against the
    // known answer:
    // * primes above 2^first_width (fold h-init leaves zero-mask slots);
    // * a single sub-chunk (fold_bits = 0: empty fold kernel);
    // * ell ≡ 1 (mod 8): a trailing width-1 sub-chunk (regression — used to
    //   panic in label::div2k via div2k(_, 0));
    // * chunk_size ∤ n: a short, padded last chunk with its own shape key;
    // * p = 2 alone (extract garbles unkeyed) and one odd prime alone (tape
    //   recorded, never replayed).
    let mut rng = rng();
    let cases: &[(&[u64], u32)] = &[
        (&[251, 257, 263], 16),  // p > 2^8 = 2^first_width
        (&[3, 5, 7], 4),         // single sub-chunk, fold_bits = 0
        (&[3, 5, 7, 11, 13], 8), // ell = 9 → sub_widths [8, 1]
        (&[3, 5, 7, 11, 13], 9), // cs = 4, chunks [4, 4, 1]: short last chunk
        (&[2], 4),               // p = 2 only: no tape ever recorded
        (&[13], 6),              // one odd prime: tape recorded, never replayed
    ];
    for &(primes, n) in cases {
        let params = CrtParams::from_primes(primes, n);
        let m: u128 = primes.iter().map(|&p| p as u128).product();
        for _ in 0..3 {
            let s_dim = 3usize;
            let a_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..m.min(u64::MAX as u128) as u64))
                .collect();
            let b_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..m.min(u64::MAX as u128) as u64))
                .collect();
            let x: u64 = rng.random_range(0..(1u64 << n));
            assert_s_aff_streaming_correct(&params, &a_vals, &b_vals, x, &mut rng);
        }
    }
}

#[test]
fn test_fold_kernel_cost_matches_system_fold() {
    // The fold kernel's cost ledger must charge exactly what the System path
    // charged for the same circuit: per prime, `fold_to_mod_ohe` adds
    // `fold_bits` CF Z₂ joins and `fold_bits·p` CF Z₂-payload switches. Build
    // the full-fold System and the extract-only System and assert
    // the difference equals FoldCost, for several (ell, p).
    use crate::comp_gc::convert::{compute_sub_widths, fold_to_mod_ohe, sub_chunk_extract};
    use crate::comp_gc::fold::fold_batch_garble;
    use crate::pipeline::sample_cf_mask;

    let mut rng = rng();
    for (ell, primes) in [(11u32, vec![2u64, 5, 29]), (22, vec![2u64, 257, 409])] {
        let sub_widths = compute_sub_widths(ell, 8);
        let first_width = sub_widths[0];
        let fold_bits: usize = sub_widths[1..].iter().map(|&w| w as usize).sum();
        for &p in &primes {
            let build = |with_fold: bool| {
                let mut sys = System::new();
                let r = sys.input_bits(ell);
                let ex = sub_chunk_extract(&mut sys, r, &sub_widths);
                if with_fold {
                    fold_to_mod_ohe(&mut sys, &ex, p);
                }
                sys.cost()
            };
            let old = build(true);
            let new = build(false);
            let fbh_masks: Vec<_> = (0..1usize << first_width)
                .map(|_| sample_cf_mask(&mut rng, 2))
                .collect();
            let bit_masks: Vec<_> = (0..fold_bits)
                .map(|_| sample_cf_mask(&mut rng, 2))
                .collect();
            let g = fold_batch_garble(p, &fbh_masks, &bit_masks, first_width, 0);
            assert_eq!(
                old.join_complexity_cf,
                new.join_complexity_cf + g.cost.join_complexity_cf,
                "CF join parity broke for ell={ell}, p={p}"
            );
            assert_eq!(
                old.hash_count_cf,
                new.hash_count_cf + g.cost.hash_count_cf,
                "CF hash parity broke for ell={ell}, p={p}"
            );
            assert_eq!(old.join_complexity_ncf, new.join_complexity_ncf);
            assert_eq!(old.hash_count_ncf, new.hash_count_ncf);
        }
    }
}

/// Fast non-cryptographic PRG (SplitMix64) for benchmarks, so share/randomness
/// sampling does not dominate the measurement of garbling compute. Stands in for
/// the garbler's local PRG / random-oracle expansion of randomness. (The switch
/// system draws its garbling randomness from CCRH/AES, already counted in its
/// time, so this only affects bit-decomp share sampling.)
struct BenchRng {
    state: u64,
}

impl BenchRng {
    fn new(seed: u64) -> Self {
        BenchRng { state: seed }
    }
}

impl rand::RngCore for BenchRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut i = 0;
        while i < dst.len() {
            let r = self.next_u64().to_le_bytes();
            let n = (dst.len() - i).min(8);
            dst[i..i + n].copy_from_slice(&r[..n]);
            i += n;
        }
    }
}

/// Final head-to-head benchmark: switch system (CRT) vs the bit-decomposition
/// baseline ([`crate::bitdecomp`]) on the SAME `a·x+b` workload, on identical
/// footing — single-threaded, release, warmup + ITERS iterations, MEDIAN [MIN]
/// reported. Both EXCLUDE a,b sampling (done once before timing) and any
/// correctness/decode step (correctness is covered by
/// `bitdecomp::tests::test_garble_eval_matches_plaintext` and the assert in
/// `test_s_aff_scaling`). Input-label provisioning is outside the timed
/// region for both (bit-decomp `encode`; the switch driver samples input
/// masks outside its per-stage timers). The switch side runs the kernel
/// production path ([`crate::affine::build_s_aff_kernels`]).
///
/// Workload boundary: the garbler STARTS from (a, b) over the field and the
/// evaluator ENDS with a·x+b over the field — so the switch garble column
/// includes CRT-encoding a, b into per-prime residues, and the switch eval
/// column includes the Garner reconstruction of every component. (Garner
/// setup is per-prime-set precompute, outside; statistical smudging of b is
/// parameter preparation per the `build_s_aff_streaming` docs and excluded —
/// it would add ~25-90 ms at S=76,800 depending on the μ-PRG.)
///
/// Run:
///   cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
/// Override via env: `N=256 S=1536 ITERS=30 WARMUP=5 cargo test -r ...`
#[test]
#[ignore]
fn bench_axb_comparison() {
    use crate::bitdecomp;
    use std::hint::black_box;
    use std::time::Instant;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let n = env("N", 256) as u32;
    let s = env("S", 1536);
    let iters = env("ITERS", 30);
    let warmup = env("WARMUP", 5);
    assert_eq!(
        n as usize,
        bitdecomp::N_BITS,
        "bit-decomp N_BITS is a compile-time constant ({}); set N to match",
        bitdecomp::N_BITS
    );

    // (min_ms, median_ms) from per-iteration seconds.
    fn stats(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0] * 1e3, v[v.len() / 2] * 1e3)
    }

    // Fast PRG for both sides: bit-decomp share sampling uses it directly; the
    // switch system only uses it for out-of-band (untimed) Δ / input-mask draws.
    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ===================== bit-decomposition baseline =====================
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x_fp = bitdecomp::Fp::random(&mut rng);

    let mut bd_g = Vec::with_capacity(iters);
    let mut g_keep = None;
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(&g);
        if it >= warmup {
            bd_g.push(dt);
        }
        g_keep = Some(g);
    }
    let g = g_keep.unwrap();
    let active = bitdecomp::encode(&g, x_fp);
    let mut bd_e = Vec::with_capacity(iters);
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let out = bitdecomp::eval(&g, &active, x_fp, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(out);
        if it >= warmup {
            bd_e.push(dt);
        }
    }
    let (bdg_min, bdg_med) = stats(bd_g);
    let (bde_min, bde_med) = stats(bd_e);
    let bd_h = bitdecomp::cost_model(s);
    let (bd_comm, bd_comm_grr) = (g.comm_bytes(), g.comm_bytes_grr());

    // ===================== switch system (CRT) =====================
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let decoder = GarnerDecoder::new(&params.primes); // per-prime-set precompute

    let mut sw_g = Vec::with_capacity(iters);
    let mut sw_e = Vec::with_capacity(iters);
    let mut sw_t = Vec::with_capacity(iters);
    let (mut sw_cf, mut sw_ncf, mut sw_comm_bits) = (0usize, 0usize, 0usize);
    let (mut sw_enc_ms, mut sw_dec_ms) = (0.0f64, 0.0f64);
    for it in 0..warmup + iters {
        let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();

        // Garbler-side CRT encode: (a, b) over the field -> per-prime residues.
        let t = Instant::now();
        let a_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
            .collect();
        let b_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
            .collect();
        let t_enc = t.elapsed().as_secs_f64();

        // The kernel driver samples input masks internally but outside its
        // per-stage timers, so garble/eval exclude label provisioning —
        // matching bit-decomp's untimed encode().
        let t = Instant::now();
        let (out, stats) =
            crate::affine::build_s_aff_kernels(&mut rng, &x_bits, &params, &a_res, &b_res);
        let core = t.elapsed().as_secs_f64();
        black_box(&out);

        // Evaluator-side CRT decode: Garner-reconstruct every component to
        // its field value.
        let t = Instant::now();
        let mut residues = vec![0u64; params.primes.len()];
        for j in 0..s {
            for (i, prime_outs) in out.iter().enumerate() {
                residues[i] = prime_outs[j];
            }
            black_box(decoder.reconstruct(&residues));
        }
        let t_dec = t.elapsed().as_secs_f64();

        if it >= warmup {
            sw_g.push(stats.garble_secs() + t_enc);
            sw_e.push(stats.eval_secs() + t_dec);
            sw_t.push(t_enc + core + t_dec);
            sw_cf = stats.hash_count_cf;
            sw_ncf = stats.hash_count_ncf;
            sw_comm_bits =
                crate::label::LAMBDA * stats.join_complexity_cf + stats.join_complexity_ncf;
            sw_enc_ms = t_enc * 1e3;
            sw_dec_ms = t_dec * 1e3;
        }
    }
    let (swg_min, swg_med) = stats(sw_g);
    let (swe_min, swe_med) = stats(sw_e);
    let (swt_min, swt_med) = stats(sw_t);

    // ===================== report =====================
    let mib = |b: usize| b as f64 / 1_048_576.0;
    let mb = |bits: usize| bits as f64 / 8.0 / 1e6;
    let m = |x: usize| x as f64 / 1e6;
    eprintln!("\n========== a·x + b benchmark  (N={n}, S={s}, single-thread, release) ==========");
    eprintln!("iters={iters}, warmup={warmup}; values are MEDIAN [MIN] ms; switch garble incl. CRT-encode of a,b,");
    eprintln!("switch eval incl. Garner reconstruction to field values; excludes a,b sampling & verify\n");
    eprintln!("{:<22}{:>17}{:>17}{:>17}", "approach", "garble", "eval", "garble+eval");
    eprintln!("{:-<73}", "");
    eprintln!(
        "{:<22}{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]",
        "bit-decomposition", bdg_med, bdg_min, bde_med, bde_min, bdg_med + bde_med, bdg_min + bde_min
    );
    eprintln!(
        "{:<22}{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]",
        "switch system (CRT)", swg_med, swg_min, swe_med, swe_min, swg_med + swe_med, swg_min + swe_min
    );
    eprintln!(
        "{:<22}{:>15.2}x{:>16.2}x{:>16.2}x",
        "ratio switch/bitdec",
        swg_med / bdg_med,
        swe_med / bde_med,
        (swg_med + swe_med) / (bdg_med + bde_med)
    );
    eprintln!(
        "\nswitch full run: {:.2} [{:.2}] ms  (encode+seed+garble+eval+decode+other)",
        swt_med, swt_min
    );
    eprintln!(
        "switch CRT split: encode a,b {:.1} ms (in garble col) | Garner reconstruct {:.1} ms (in eval col)",
        sw_enc_ms, sw_dec_ms
    );
    eprintln!("\nhashes (CCRH 16-byte blocks, per party):");
    eprintln!(
        "   bit-decomp   garbler {:.2}M / evaluator {:.2}M   (exact: 4nS / 2nS)",
        m(bd_h.garble_calls),
        m(bd_h.eval_calls),
    );
    eprintln!(
        "   switch       garbler {:.2}M / evaluator \u{2248}{:.2}M  (cf {} + ncf {}; evaluator ~0.3% less —",
        m(sw_cf + sw_ncf),
        m(sw_cf + sw_ncf),
        sw_cf,
        sw_ncf
    );
    eprintln!("                skips open switches; exact split via `bench_axb_hashcounts`)");
    eprintln!(
        "comm:    bit-decomp  {:.1} MiB (GRR, materialized; {:.1} MiB if both rows sent)   |   switch  {:.2} MB (join width)  => ~{:.0}x less\n",
        mib(bd_comm_grr),
        mib(bd_comm),
        mb(sw_comm_bits),
        mib(bd_comm_grr) / (mb(sw_comm_bits) / 1.048576)
    );
}

/// End-to-end latency benchmark: garble → transmit the garbled material over a
/// simulated bandwidth-limited link → eval, for both approaches on the same
/// a·x+b workload. Transmission is modeled analytically as
/// `(bytes · 8) / bandwidth` (pure serialization delay, no RTT); only garble,
/// transmission, and eval are counted — no build/setup, no a,b sampling, no
/// decode. Single-threaded. This is where the switch system's ~50–100× smaller
/// communication flips the comparison on a constrained link.
///
/// Run:
///   cargo test -r --lib bench_axb_network -- --ignored --nocapture
/// Override via env: `N=256 S=1536 ITERS=30 WARMUP=5 BW_MBPS=100 cargo test -r ...`
#[test]
#[ignore]
fn bench_axb_network() {
    use crate::bitdecomp;
    use std::hint::black_box;
    use std::time::Instant;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let n = env("N", 256) as u32;
    let s = env("S", 1536);
    let iters = env("ITERS", 30);
    let warmup = env("WARMUP", 5);
    let bw_mbps: f64 = std::env::var("BW_MBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100.0);
    assert_eq!(
        n as usize,
        bitdecomp::N_BITS,
        "bit-decomp N_BITS is a compile-time constant ({})",
        bitdecomp::N_BITS
    );

    // Transmission latency (ms) to push `bytes` over a `bw_mbps` link.
    let tx_ms = |bytes: usize| -> f64 { (bytes as f64) * 8.0 / (bw_mbps * 1e6) * 1e3 };
    // Median of per-iteration seconds, in ms.
    fn med(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] * 1e3
    }

    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ---- bit-decomposition ----
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x_fp = bitdecomp::Fp::random(&mut rng);
    let mut bdg = Vec::with_capacity(iters);
    let mut g_keep = None;
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(&g);
        if it >= warmup {
            bdg.push(dt);
        }
        g_keep = Some(g);
    }
    let g = g_keep.unwrap();
    let active = bitdecomp::encode(&g, x_fp);
    let mut bde = Vec::with_capacity(iters);
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        black_box(bitdecomp::eval(&g, &active, x_fp, &mut l));
        let dt = t.elapsed().as_secs_f64();
        if it >= warmup {
            bde.push(dt);
        }
    }
    let (bd_g, bd_e) = (med(bdg), med(bde));
    let bd_tx = tx_ms(g.comm_bytes_grr()); // realistic transmitted GC (one row/bit via GRR)
    let bd_e2e = bd_g + bd_tx + bd_e;

    // ---- switch system (CRT) ----
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let decoder = GarnerDecoder::new(&params.primes); // per-prime-set precompute
    let mut swg = Vec::with_capacity(iters);
    let mut swe = Vec::with_capacity(iters);
    let (mut sw_comm_bits, mut sw_cf, mut sw_ncf) = (0usize, 0usize, 0usize);
    for it in 0..warmup + iters {
        let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();
        // Garbler starts from (a, b) over the field: CRT encode is garble-side.
        let t = Instant::now();
        let a_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
            .collect();
        let b_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
            .collect();
        let t_enc = t.elapsed().as_secs_f64();
        let (out, stats) =
            crate::affine::build_s_aff_kernels(&mut rng, &x_bits, &params, &a_res, &b_res);
        black_box(&out);
        // Evaluator ends with a·x+b over the field: Garner decode is eval-side.
        let t = Instant::now();
        let mut residues = vec![0u64; params.primes.len()];
        for j in 0..s {
            for (i, prime_outs) in out.iter().enumerate() {
                residues[i] = prime_outs[j];
            }
            black_box(decoder.reconstruct(&residues));
        }
        let t_dec = t.elapsed().as_secs_f64();
        if it >= warmup {
            swg.push(stats.garble_secs() + t_enc);
            swe.push(stats.eval_secs() + t_dec);
            sw_comm_bits =
                crate::label::LAMBDA * stats.join_complexity_cf + stats.join_complexity_ncf;
            sw_cf = stats.hash_count_cf;
            sw_ncf = stats.hash_count_ncf;
        }
    }
    let (sw_g, sw_e) = (med(swg), med(swe));
    let sw_bytes = sw_comm_bits / 8;
    let sw_tx = tx_ms(sw_bytes);
    let sw_e2e = sw_g + sw_tx + sw_e;

    // ---- report ----
    let mib = |b: usize| b as f64 / 1_048_576.0;
    eprintln!(
        "\n===== a·x + b end-to-end over a {:.0} Mbps link  (N={n}, S={s}, single-thread, release) =====",
        bw_mbps
    );
    eprintln!("iters={iters}, warmup={warmup}; garble/eval = MEDIAN ms; transmit = (GC bytes·8)/bandwidth\n");
    eprintln!(
        "{:<22}{:>10}{:>16}{:>10}{:>14}",
        "approach", "garble", "transmit", "eval", "END-TO-END"
    );
    eprintln!("{:-<72}", "");
    eprintln!(
        "{:<22}{:>7.2}ms{:>9.1}ms{:>14}{:>7.2}ms{:>11.1}ms",
        "bit-decomposition",
        bd_g,
        bd_tx,
        format!("({:.1} MiB)", mib(g.comm_bytes_grr())),
        bd_e,
        bd_e2e
    );
    eprintln!(
        "{:<22}{:>7.2}ms{:>9.1}ms{:>14}{:>7.2}ms{:>11.1}ms",
        "switch system (CRT)",
        sw_g,
        sw_tx,
        format!("({:.2} MB)", sw_bytes as f64 / 1e6),
        sw_e,
        sw_e2e
    );
    eprintln!(
        "\n=> switch system is {:.1}x FASTER end-to-end ({:.1} ms vs {:.0} ms): tiny comm dominates\n   the bandwidth term that the bit-decomposition GC ({:.1} MiB) pays.",
        bd_e2e / sw_e2e,
        sw_e2e,
        bd_e2e,
        mib(g.comm_bytes_grr())
    );
    eprintln!(
        "   (bit-decomp without GRR sends {:.1} MiB => transmit {:.1} ms, end-to-end {:.1} ms.)",
        mib(g.comm_bytes()),
        tx_ms(g.comm_bytes()),
        bd_g + tx_ms(g.comm_bytes()) + bd_e
    );
    let bd_h = bitdecomp::cost_model(s);
    let mh = |x: usize| x as f64 / 1e6;
    eprintln!("\nhashes (CCRH 16-byte blocks, per party):");
    eprintln!(
        "   bit-decomp   garbler {:.2}M / evaluator {:.2}M   (exact: 4nS / 2nS)",
        mh(bd_h.garble_calls),
        mh(bd_h.eval_calls),
    );
    eprintln!(
        "   switch       garbler {:.2}M / evaluator \u{2248}{:.2}M  (cf {} + ncf {}; evaluator ~0.3% less,",
        mh(sw_cf + sw_ncf),
        mh(sw_cf + sw_ncf),
        sw_cf,
        sw_ncf
    );
    eprintln!("                skips open switches; exact split via `bench_axb_hashcounts`)\n");
}

/// Exact per-party CCRH hash-call counts (16-byte AES blocks) for both
/// approaches, measured via the process-global counter. The garbler and
/// evaluator differ for the switch system: the garbler hashes every switch, but
/// the evaluator skips OPEN switches (control ≠ 0), so eval < garble. Requires
/// the `count-hashes` feature (zero-overhead counter, so this never perturbs the
/// timing benches).
///
/// Run:
///   cargo test -r --features count-hashes --lib bench_axb_hashcounts -- --ignored --nocapture
#[cfg(feature = "count-hashes")]
#[test]
#[ignore]
fn bench_axb_hashcounts() {
    use crate::bitdecomp;
    use crate::crypto::{hash_blocks, reset_hash_blocks};

    let n = 256u32;
    let s: usize = std::env::var("S").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
    assert_eq!(n as usize, bitdecomp::N_BITS);
    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ---- bit-decomposition: garbler vs evaluator blocks ----
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x = bitdecomp::Fp::random(&mut rng);
    let mut led = bitdecomp::HashLedger::default();
    reset_hash_blocks();
    let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut led);
    let bd_g = hash_blocks();
    let active = bitdecomp::encode(&g, x);
    reset_hash_blocks();
    let _ = bitdecomp::eval(&g, &active, x, &mut led);
    let bd_e = hash_blocks();

    // ---- switch system: per-party blocks accumulated across the stream ----
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let a_res: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_res: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();
    let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();
    reset_hash_blocks();
    let (_, stats) =
        crate::affine::build_s_aff_kernels(&mut rng, &x_bits, &params, &a_res, &b_res);
    let sw_total = hash_blocks();
    let sw_g = stats.garble_hash_blocks;
    let sw_e = stats.eval_hash_blocks;
    let sw_ledger = (stats.hash_count_cf + stats.hash_count_ncf) as u64;

    // Cross-checks: measured == ground truth.
    assert_eq!(bd_g as usize, led.garble_calls, "bit-decomp garbler vs HashLedger");
    assert_eq!(bd_e as usize, led.eval_calls, "bit-decomp evaluator vs HashLedger");
    assert_eq!(sw_g + sw_e, sw_total, "switch per-party blocks must sum to measured total");

    let mm = |x: u64| x as f64 / 1e6;
    eprintln!("\n===== a·x+b CCRH hash calls (16-byte AES blocks), per party  (N={n}, S={s}) =====\n");
    eprintln!("{:<22}{:>14}{:>14}{:>12}", "approach", "GARBLER", "EVALUATOR", "eval/garble");
    eprintln!("{:-<62}", "");
    eprintln!(
        "{:<22}{:>14}{:>14}{:>11.2}x",
        "bit-decomposition", bd_g, bd_e, bd_e as f64 / bd_g as f64
    );
    eprintln!(
        "{:<22}{:>14}{:>14}{:>11.2}x",
        "switch system (CRT)", sw_g, sw_e, sw_e as f64 / sw_g as f64
    );
    eprintln!(
        "\nbit-decomp closed form : garbler 4nS = {}, eval 2nS = {}  (exact, input-independent)",
        2 * bitdecomp::N_BITS * s * bitdecomp::BLOCKS_PER_FE,
        bitdecomp::N_BITS * s * bitdecomp::BLOCKS_PER_FE
    );
    eprintln!(
        "switch                 : garbler {:.2}M / evaluator {:.2}M  (analytical circuit ledger cf {} + ncf {} = {})",
        mm(sw_g),
        mm(sw_e),
        stats.hash_count_cf,
        stats.hash_count_ncf,
        sw_ledger
    );
    eprintln!(
        "  evaluator < garbler because the evaluator skips OPEN switches; the exact\n\
         evaluator count is mildly input-dependent (which one-hot slots are active).\n"
    );
}

// ==================== the kernel-only path ====================

/// Kernel-path analogue of [`assert_s_aff_streaming_correct`]: known-answer
/// check against the direct oracle (the System-path differential lives in
/// `test_s_aff_kernels_matches_streaming`).
fn assert_s_aff_kernels_correct(
    params: &CrtParams,
    a_vals: &[u64],
    b_vals: &[u64],
    x: u64,
    rng: &mut impl Rng,
) {
    let n = params.n as usize;
    let s_dim = a_vals.len();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();
    let input_bits: Vec<u64> = (0..n).map(|j| (x >> j) & 1).collect();

    let (outputs, _stats) =
        crate::affine::build_s_aff_kernels(rng, &input_bits, params, &a_residues, &b_residues);

    for (i, &p_i) in params.primes.iter().enumerate() {
        let x_mod = x % p_i;
        for j in 0..s_dim {
            let expected = ((a_vals[j] % p_i) * x_mod + (b_vals[j] % p_i)) % p_i;
            assert_eq!(
                outputs[i][j], expected,
                "kernel a·x+b mod p mismatch (prime {p_i}, comp {j}, x={x}, S={s_dim})"
            );
        }
    }
}

#[test]
fn test_s_aff_kernels_sweep() {
    // The kernel-only production path against the oracle, across the same
    // parameter sweep as test_streaming_sweep.
    let mut rng = rng();
    for n in [8u32, 12, 16, 24, 33, 64] {
        let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
        let max_x = if n >= 63 { u64::MAX } else { (1u64 << n) - 1 };
        for s_dim in [1usize, 3] {
            let a_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let b_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let x = rng.random_range(0..=max_x) & max_x;
            assert_s_aff_kernels_correct(&params, &a_vals, &b_vals, x, &mut rng);
        }
    }
}

#[test]
fn test_s_aff_kernels_matches_streaming() {
    // Differential: kernel path and System-phase path decode identical
    // outputs on identical inputs (production params, several x).
    let mut rng = rng();
    let n = 256u32;
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let s_dim = 5usize;
    let a_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();

    for _ in 0..3 {
        let input_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();

        let (kernel_out, stats) = crate::affine::build_s_aff_kernels(
            &mut rng,
            &input_bits,
            &params,
            &a_residues,
            &b_residues,
        );

        let mut pipeline = Pipeline::new(&mut rng);
        let bit_ids: Vec<_> = input_bits
            .iter()
            .map(|&b| pipeline.seed_input_cf_value(&mut rng, 2, b))
            .collect();
        let stream_out = build_s_aff_streaming(
            &mut pipeline,
            &bit_ids,
            &input_bits,
            &params,
            &a_residues,
            &b_residues,
        );

        assert_eq!(kernel_out, stream_out, "kernel vs streaming outputs differ");
        // Ledger parity: the two paths build cost-identical circuit
        // equivalents (labels, masks, and nonce layouts differ by
        // construction — only decoded outputs and the ledger must agree).
        assert_eq!(stats.hash_count_cf, pipeline.hash_count_cf);
        assert_eq!(stats.hash_count_ncf, pipeline.hash_count_ncf);
        assert_eq!(stats.join_complexity_cf, pipeline.join_complexity_cf);
        assert_eq!(stats.join_complexity_ncf, pipeline.join_complexity_ncf);
        // Both garblers hash every switch (x-blind), so the measured
        // per-party block counts agree exactly too (kernel G == streaming G
        // == cf + ncf ledger) — but the `count-hashes` counter is
        // process-global and races with parallel tests, so that equality is
        // only checked in the single-run benches, not asserted here.
    }
}

#[test]
#[ignore]
fn bench_kernel_loop() {
    // Kernel-path twin of bench_stream_loop.
    // Run: N=256 S=1536 REPS=30 cargo test --release bench_kernel_loop -- --ignored --nocapture
    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "256".into())
        .parse()
        .unwrap();
    let s_dim: usize = std::env::var("S")
        .unwrap_or_else(|_| "1536".into())
        .parse()
        .unwrap();
    let reps: usize = std::env::var("REPS")
        .unwrap_or_else(|_| "30".into())
        .parse()
        .unwrap();
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let mut rng = rng();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let mut agg = crate::affine::KernelStats::default();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let x: u64 = rng.random_range(0..u64::MAX >> (64 - n.min(63)));
        let x_bits: Vec<u64> = (0..n).map(|j| (x >> (j % 64)) & 1).collect();
        let (outputs, stats) = crate::affine::build_s_aff_kernels(
            &mut rng,
            &x_bits,
            &params,
            &a_residues,
            &b_residues,
        );
        std::hint::black_box(&outputs);
        agg.chunk_garble_secs += stats.chunk_garble_secs;
        agg.chunk_eval_secs += stats.chunk_eval_secs;
        agg.extract_garble_secs += stats.extract_garble_secs;
        agg.extract_eval_secs += stats.extract_eval_secs;
        agg.fold_garble_secs += stats.fold_garble_secs;
        agg.fold_eval_secs += stats.fold_eval_secs;
        agg.body_garble_secs += stats.body_garble_secs;
        agg.body_eval_secs += stats.body_eval_secs;
    }
    let per_rep = t.elapsed().as_secs_f64() / reps as f64;
    let r = reps as f64;
    eprintln!("{} reps, {:.4}s/rep (kernel path)", reps, per_rep);
    eprintln!(
        "  per-rep garble: chunk {:.2}ms extract {:.2}ms fold {:.2}ms body {:.2}ms",
        1e3 * agg.chunk_garble_secs / r,
        1e3 * agg.extract_garble_secs / r,
        1e3 * agg.fold_garble_secs / r,
        1e3 * agg.body_garble_secs / r,
    );
    eprintln!(
        "  per-rep eval:   chunk {:.2}ms extract {:.2}ms fold {:.2}ms body {:.2}ms",
        1e3 * agg.chunk_eval_secs / r,
        1e3 * agg.extract_eval_secs / r,
        1e3 * agg.fold_eval_secs / r,
        1e3 * agg.body_eval_secs / r,
    );
    eprintln!(
        "  accounted {:.2}ms of {:.2}ms/rep ({:.0}%)",
        1e3 * (agg.garble_secs() + agg.eval_secs()) / r,
        1e3 * per_rep,
        100.0 * (agg.garble_secs() + agg.eval_secs()) / r / per_rep,
    );
}

#[test]
fn test_s_aff_kernels_edge_regimes() {
    // Kernel-path twin of test_streaming_edge_regimes, minus the width-1
    // trailing sub-chunk (ell ≡ 1 mod 8), which the kernel path rejects by
    // design (covered by test_extract_kernel_rejects_width1; use the
    // reference path for those shapes).
    let mut rng = rng();

    // p > 2^first_width and p = 2 alone, one odd prime, short padded chunk.
    for primes in [&[2u64][..], &[409][..], &[2, 3, 5, 7, 401, 409][..]] {
        let params = CrtParams::from_primes(primes, 16);
        let a = vec![rng.random_range(0..1u64 << 30)];
        let b = vec![rng.random_range(0..1u64 << 30)];
        let x = rng.random_range(0..1u64 << 16);
        assert_s_aff_kernels_correct(&params, &a, &b, x, &mut rng);
    }
    // Single sub-chunk (ell <= 8): tiny prime set keeps the sum small.
    let params = CrtParams::from_primes(&[3u64, 5], 4);
    assert!(params.ell <= 8, "want a single-sub-chunk regime");
    for x in 0..16u64 {
        assert_s_aff_kernels_correct(&params, &[7], &[11], x, &mut rng);
    }
}
