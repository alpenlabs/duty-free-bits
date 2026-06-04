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
use crate::crt::{CrtParams, crt_reconstruct};
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
/// the production path; this needs no reference implementation).
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
    let outputs =
        build_s_aff_streaming(&mut pipeline, &bit_ids, &input_bits, params, &a_residues, &b_residues);

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
    let a_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..1u64 << 48)).collect();
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

        // Verify outputs reconstruct correctly.
        for s in 0..s_dim {
            let residues: Vec<u64> = outputs.iter().map(|prime_outs| prime_outs[s]).collect();
            let reconstructed = crt_reconstruct(&residues, &params.primes);
            let expected = a_vals[s] * x + b_vals[s];
            assert_eq!(reconstructed, U576::from_u64(expected));
        }
        eprintln!("ok: streaming reconstructed all {s_dim} affine maps");
    }
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
