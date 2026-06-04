//! Integration tests for the switch system, execution engine, and affine maps.
//!
//! Unit tests for individual modules live in their respective `#[cfg(test)]` blocks:
//! - `types.rs`: Val arithmetic
//! - `components/ohe.rs`: one-hot encoding
//! - `components/convert.rs`: bin_to_word, word_to_hot, word_to_ring, hot_to_ring
//! - `components/crt.rs`: CRT parameters and reconstruction
//! - `components/bigint.rs`: U576 arithmetic

use crate::components::affine::{build_s_aff, build_s_aff_streaming};
use crate::components::bigint::{FIRST_80_PRIMES, U576};
use crate::components::crt::{CrtParams, crt_reconstruct};
use crate::exec::Exec;
use crate::garble::label::{CfLabel, LAMBDA};
use crate::garble::{Label, Pipeline, Program, eval, garble, normalize_delta};
use crate::system::System;
use crate::types::*;

use rand::Rng;

/// Sample a uniform CF mask for a power-of-two modulus.
fn sample_cf_mask<R: Rng>(rng: &mut R, modulus: u64) -> Label {
    assert!(modulus.is_power_of_two());
    let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
    Label::Cf(CfLabel::from_coords(&coords, modulus))
}

/// Run the full garble→eval round-trip on the given system and check that the
/// evaluator recovers exactly the `Exec` outputs. Returns the program so the
/// caller can assert on its size.
fn garble_eval_roundtrip<R: Rng>(
    rng: &mut R,
    sys: &System,
    input_wires: &[Wire],
    input_values: &[Val],
    output_wires: &[Wire],
    exec_outputs: &[Val],
) -> Program {
    let delta: u128 = normalize_delta(rng.random());
    let input_masks: Vec<Label> = input_wires
        .iter()
        .map(|&w| sample_cf_mask(rng, sys.modulus(w)))
        .collect();
    let program = garble(sys, input_wires, &input_masks, output_wires, delta);
    let got = eval(
        sys,
        input_wires,
        input_values,
        delta,
        &input_masks,
        &program,
        output_wires,
    );
    assert_eq!(got.len(), exec_outputs.len());
    for (i, (g, e)) in got.iter().zip(exec_outputs.iter()).enumerate() {
        assert_eq!(g, e, "output {} mismatch: got {} expected {}", i, g, e);
    }
    program
}

const SAMPLES: usize = 10;
const MOD: u64 = 16;

fn rng() -> impl Rng {
    rand::rng()
}

/// Generate a random U576 uniformly in [0, bound) via rejection sampling.
fn rand_u576_below(rng: &mut impl Rng, bound: &U576) -> U576 {
    let top = bound.0.iter().rposition(|&l| l != 0).unwrap();
    let top_bits = 64 - bound.0[top].leading_zeros();
    loop {
        let mut limbs = [0u64; 9];
        for limb in limbs.iter_mut().take(top) {
            *limb = rng.random();
        }
        limbs[top] = if top_bits < 64 {
            rng.random::<u64>() & ((1u64 << top_bits) - 1)
        } else {
            rng.random()
        };
        let candidate = U576(limbs);
        if candidate < *bound {
            return candidate;
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
    assert_eq!(sys.join_complexity, 3); // log2(8) = 3
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
fn test_s_aff_s3_primorial_10() {
    // M = 2·3·5·7·11·13·17·19·23·29, S=3, random (a, b, x)
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let params = CrtParams::from_primes(&primes, 20);
    assert_eq!(params.primorial().to_u128(), Some(6469693230));

    let n = params.n;
    let mut rng = rng();

    let m = params.primorial().to_u128().unwrap();
    let max_x = 1u64 << n;

    for _ in 0..SAMPLES {
        let a_vals: Vec<u64> = (0..3).map(|_| rng.random_range(0..m as u64)).collect();
        let b_vals: Vec<u64> = (0..3).map(|_| rng.random_range(0..m as u64)).collect();
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

        let mut sys = System::new();
        let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
        let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);

        let mut exec = Exec::new(&sys);
        for j in 0..n {
            exec.set(bits[j as usize], Val::new((x >> j) & 1, 2));
        }
        exec.run();

        for s in 0..3 {
            let residues: Vec<u64> = result
                .outputs
                .iter()
                .map(|prime_outs| exec.get(prime_outs[s]).v)
                .collect();
            let reconstructed = crt_reconstruct(&residues, &params.primes);
            let expected = ((a_vals[s] as u128) * (x as u128) + (b_vals[s] as u128)) % m;
            assert_eq!(
                reconstructed.to_u128().unwrap(),
                expected,
                "S={s}, a={}, b={}, x={x}: got {}, expected {expected} (mod M={m})",
                a_vals[s],
                b_vals[s],
                reconstructed,
            );
        }
    }
}

#[test]
fn test_s_aff_s1_primorial_80() {
    // Full 80-prime CRT pipeline with target parameters: n=256, ell=22.
    // Coefficients a, b are random elements of Z_M (M ≈ 2^553), x is 256 bits.
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, 256);
    let n = params.n as usize;
    let m = params.primorial();
    let mut rng = rng();

    for _ in 0..3 {
        // Pick random a, b ∈ [0, M), then derive CRT residues.
        let a = rand_u576_below(&mut rng, &m);
        let b = rand_u576_below(&mut rng, &m);

        let a_residues: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| vec![a.mod_u64(pi)])
            .collect();
        let b_residues: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| vec![b.mod_u64(pi)])
            .collect();

        // Random 256-bit input x (as individual bits).
        let x_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();

        let mut sys = System::new();
        let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
        let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);

        let mut exec = Exec::new(&sys);
        for j in 0..n {
            exec.set(bits[j], Val::new(x_bits[j], 2));
        }
        exec.run();

        // Verify each prime's output residue: (a_i · x + b_i) mod p_i
        let output_residues: Vec<u64> = result
            .outputs
            .iter()
            .map(|prime_outs| exec.get(prime_outs[0]).v)
            .collect();

        for (t, &p_t) in params.primes.iter().enumerate() {
            let mut x_mod_p = 0u64;
            let mut pow2 = 1u64;
            for &bit in &x_bits {
                x_mod_p = (x_mod_p + bit * pow2) % p_t;
                pow2 = (pow2 * 2) % p_t;
            }
            let expected = (a_residues[t][0] * x_mod_p + b_residues[t][0]) % p_t;
            assert_eq!(
                output_residues[t], expected,
                "prime {p_t}: got {}, expected {expected}",
                output_residues[t]
            );
        }

        // Verify CRT reconstruction is consistent with the residues.
        let reconstructed = crt_reconstruct(&output_residues, &params.primes);
        for (t, &p_t) in params.primes.iter().enumerate() {
            assert_eq!(
                reconstructed.mod_u64(p_t),
                output_residues[t],
                "CRT consistency failed for prime {p_t}"
            );
        }
    }
}

#[test]
#[ignore]
fn test_s_aff_scaling() {
    // Run with: N=12 S=4 GARBLE=1 /usr/bin/time -l cargo test --release \
    //   test_s_aff_scaling -- --ignored --nocapture
    //
    // Env vars:
    //   N        — input bit-length (default 8)
    //   S        — number of affine maps (default 1)
    //   GARBLE   — if set to "1", also time the all-at-once garble + eval round-trip
    //   STREAM   — if set to "1", run the streaming pipeline (Pipeline + build_s_aff_streaming)
    use crate::system::LAMBDA_BITS;
    use std::time::Instant;

    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .expect("N must be a u32");
    let s_dim: usize = std::env::var("S")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .expect("S must be a usize");
    let do_garble: bool = std::env::var("GARBLE").as_deref() == Ok("1");
    let do_stream: bool = std::env::var("STREAM").as_deref() == Ok("1");

    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    eprintln!(
        "n={n}, S={s_dim}, ell={}, chunk_size={}, num_chunks={}, garble={do_garble}",
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

    // The all-at-once side of the test (build + Exec + optional garble+eval).
    // Skipped when STREAM=1 is set without GARBLE=1 — the streaming side does
    // its own verification, so the giant System we'd otherwise build first
    // would just inflate peak RSS with state we then discard.
    let do_all_at_once = do_garble || !do_stream;
    if do_all_at_once {
    let mut sys = System::new();
    let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();

    let t_build = Instant::now();
    let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);
    let build_secs = t_build.elapsed().as_secs_f64();
    eprintln!(
        "[build]  {:>7.2}s  | {} wires, {} gates, {} switch groups",
        build_secs,
        sys.num_wires(),
        sys.num_gates(),
        sys.num_switch_groups(),
    );
    eprintln!(
        "          join: {} bits (cf {}, ncf {})  hash: cf {}, ncf {}",
        sys.join_complexity,
        sys.join_complexity_cf,
        sys.join_complexity_ncf,
        sys.hash_count_cf,
        sys.hash_count_ncf,
    );

    // Bulk-pack rebate report.
    let naive_ncf: usize = sys
        .gates
        .iter()
        .filter(|g| matches!(g.typ, GateType::Switch) && !sys.is_cf(g.out))
        .count();
    let packed_ncf_groups: usize = (0..sys.num_switch_groups())
        .map(|i| {
            let g = sys.switch_group(i);
            let lg_m = ((g.modulus - 1).ilog2() + 1) as usize;
            (g.members.len() * lg_m).div_ceil(LAMBDA_BITS)
        })
        .sum();
    let saved = naive_ncf - sys.hash_count_ncf;
    eprintln!(
        "          bulk-pack: naive_ncf {} → packed_ncf {} ({}× rebate, saved {})",
        naive_ncf,
        sys.hash_count_ncf,
        if packed_ncf_groups > 0 {
            naive_ncf as f64 / packed_ncf_groups as f64
        } else {
            1.0
        },
        saved,
    );

    let input_values: Vec<Val> = (0..n).map(|j| Val::new((x >> j) & 1, 2)).collect();

    let t_exec = Instant::now();
    let mut exec = Exec::new(&sys);
    for (&w, &v) in bits.iter().zip(input_values.iter()) {
        exec.set(w, v);
    }
    exec.run();
    let exec_secs = t_exec.elapsed().as_secs_f64();
    eprintln!("[exec]   {:>7.2}s  | reference value propagation", exec_secs);

    // Verify Exec outputs reconstruct correctly.
    for s in 0..s_dim {
        let residues: Vec<u64> = result
            .outputs
            .iter()
            .map(|prime_outs| exec.get(prime_outs[s]).v)
            .collect();
        let reconstructed = crt_reconstruct(&residues, &params.primes);
        let expected = a_vals[s] * x + b_vals[s];
        assert_eq!(reconstructed, U576::from_u64(expected));
    }

    if do_garble {
        let mut output_wires: Vec<Wire> = Vec::with_capacity(80 * s_dim);
        for row in &result.outputs {
            output_wires.extend(row.iter().copied());
        }
        let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();

        let delta: u128 = normalize_delta(rng.random());
        let input_masks: Vec<Label> = bits
            .iter()
            .map(|&w| sample_cf_mask(&mut rng, sys.modulus(w)))
            .collect();

        let t_garble = Instant::now();
        let program = garble(&sys, &bits, &input_masks, &output_wires, delta);
        let garble_secs = t_garble.elapsed().as_secs_f64();
        eprintln!(
            "[garble] {:>7.2}s  | program: {} bits ({:.2} MB)",
            garble_secs,
            program.total_bits(),
            program.total_bits() as f64 / 8.0 / 1024.0 / 1024.0,
        );

        let t_eval = Instant::now();
        let got = eval(
            &sys,
            &bits,
            &input_values,
            delta,
            &input_masks,
            &program,
            &output_wires,
        );
        let eval_secs = t_eval.elapsed().as_secs_f64();
        eprintln!("[eval]   {:>7.2}s  | label propagation + decode", eval_secs);
        for (i, (g, e)) in got.iter().zip(exec_outputs.iter()).enumerate() {
            assert_eq!(g, e, "output {i} mismatch");
        }

        // Theoretical hashing cost at AES-NI speed (~1.5 GB/s). The CCRH stub
        // is currently zero, so this isn't *measured* time — it's the budget
        // a real CCRH would need.
        let total_hashes = sys.hash_count_cf + sys.hash_count_ncf;
        let hash_bytes = total_hashes * (LAMBDA_BITS / 8);
        let aes_ni_gbs = 1.5; // conservative AES-MMO throughput on Apple Silicon
        let est_hash_secs = (hash_bytes as f64 / 1e9) / aes_ni_gbs;
        eprintln!(
            "[hash*]  {:>7.3}s  | {} CCRH calls × λ bits = {:.1} MB at {} GB/s (estimate; current H=0)",
            est_hash_secs,
            total_hashes,
            hash_bytes as f64 / 1024.0 / 1024.0,
            aes_ni_gbs,
        );

        let total = build_secs + garble_secs + eval_secs;
        eprintln!(
            "                    | build {:.0}% | garble {:.0}% | eval {:.0}% | hash budget {:.1}%",
            100.0 * build_secs / total,
            100.0 * garble_secs / total,
            100.0 * eval_secs / total,
            100.0 * est_hash_secs / total,
        );
    }

    eprintln!("ok: all {s_dim} affine maps reconstructed");
    } // end do_all_at_once

    if do_stream {
        // Streaming path: builds + garbles + evals one phase at a time, dropping
        // intermediate state at every phase boundary.
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
        eprintln!(
            "          program: {} bits ({:.2} MB)",
            pipeline.total_program_bits,
            pipeline.total_program_bits as f64 / 8.0 / 1024.0 / 1024.0,
        );
        eprintln!(
            "          join: total {} bits (cf {}, ncf {})  hash: cf {}, ncf {}",
            pipeline.join_complexity_cf + pipeline.join_complexity_ncf,
            pipeline.join_complexity_cf,
            pipeline.join_complexity_ncf,
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

// ==================== Garble / Eval round-trip tests ====================

#[test]
fn test_garble_hot_to_ring_minimal() {
    // The smallest interesting NCF-output round-trip: evaluate identity over Z_3
    // via a 2-entry binary OHE derived from one Z_2 input.
    use crate::components::convert::hot_to_ring;

    for b in 0..2u64 {
        let mut sys = System::new();
        let bit = sys.input(2);
        let nbit = sys.not(bit);
        let h = vec![nbit, bit];
        let one_z3 = sys.constant(1, 3);
        let zero_z3 = sys.constant(0, 3);
        let table = [0u64, 1u64];
        let out = hot_to_ring(&mut sys, &h, &table, one_z3, zero_z3);

        let mut exec = Exec::new(&sys);
        exec.set(bit, Val::new(b, 2));
        exec.run();
        let expected = exec.get(out);
        assert_eq!(expected, Val::new(b, 3));

        let mut rng = rng();
        let _ = garble_eval_roundtrip(
            &mut rng,
            &sys,
            &[bit],
            &[Val::new(b, 2)],
            &[out],
            &[expected],
        );
    }
}


#[test]
fn test_garble_s_aff_s3_primorial_10() {
    // Mirror test_s_aff_s3_primorial_10 but additionally check garble→eval
    // recovers the same residues Exec would. Uses FIRST_80_PRIMES[..10] so the
    // test exercises p=2 as well (affine.rs declares its coefficient constants
    // NCF, which keeps the output labeled NCF even when the ring is Z_2).
    let primes: [u64; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let params = CrtParams::from_primes(&primes, 20);

    let n = params.n;
    let mut rng = rng();

    let m: u128 = params.primes.iter().map(|&p| p as u128).product();
    let max_x = 1u64 << n;

    for _ in 0..SAMPLES {
        let a_vals: Vec<u64> = (0..3).map(|_| rng.random_range(0..m as u64)).collect();
        let b_vals: Vec<u64> = (0..3).map(|_| rng.random_range(0..m as u64)).collect();
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

        let mut sys = System::new();
        let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
        let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);

        // Flatten outputs as (prime, component) pairs, row-major by prime then s.
        let mut output_wires: Vec<Wire> = Vec::new();
        for row in &result.outputs {
            output_wires.extend(row.iter().copied());
        }

        // Reference: Exec.
        let mut exec = Exec::new(&sys);
        let input_values: Vec<Val> =
            (0..n).map(|j| Val::new((x >> j) & 1, 2)).collect();
        for (&w, &v) in bits.iter().zip(input_values.iter()) {
            exec.set(w, v);
        }
        exec.run();
        let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();

        // Round-trip.
        let program = garble_eval_roundtrip(
            &mut rng,
            &sys,
            &bits,
            &input_values,
            &output_wires,
            &exec_outputs,
        );

        // Size sanity: program_bits ≥ join_width (the paper's lower bound; output
        // labels add on top). Exact equality is exercised in
        // test_garble_program_size_matches_join_width.
        let join_width_bits: usize = program
            .total_bits()
            .saturating_sub(expected_output_bits(&sys, &output_wires));
        assert!(
            join_width_bits > 0,
            "expected non-trivial join width for primorial-10 run"
        );

        // Reconstruct for each of the 3 components and check against expected.
        for s in 0..3 {
            let residues: Vec<u64> = (0..params.num_primes)
                .map(|i| exec_outputs[i * 3 + s].v)
                .collect();
            let reconstructed = crt_reconstruct(&residues, &params.primes);
            let expected = ((a_vals[s] as u128) * (x as u128) + (b_vals[s] as u128)) % m;
            assert_eq!(
                reconstructed.to_u128().unwrap(),
                expected,
                "S={s}, a={}, b={}, x={x}",
                a_vals[s],
                b_vals[s]
            );
        }
    }
}

#[test]
fn test_garble_s_aff_s1_primorial_80() {
    // Full 80-prime pipeline (includes p=2). Works because affine.rs declares
    // its coefficient constants NCF, so the Z_2 residue wire is NCF too.
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, 256);
    let n = params.n as usize;
    let m = params.primorial();
    let mut rng = rng();

    for _ in 0..2 {
        let a = rand_u576_below(&mut rng, &m);
        let b = rand_u576_below(&mut rng, &m);

        let a_residues: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| vec![a.mod_u64(pi)])
            .collect();
        let b_residues: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| vec![b.mod_u64(pi)])
            .collect();

        let x_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();

        let mut sys = System::new();
        let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
        let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);

        let output_wires: Vec<Wire> = result
            .outputs
            .iter()
            .map(|row| row[0])
            .collect();

        // Reference outputs via Exec.
        let input_values: Vec<Val> =
            x_bits.iter().map(|&b| Val::new(b, 2)).collect();
        let mut exec = Exec::new(&sys);
        for (&w, &v) in bits.iter().zip(input_values.iter()) {
            exec.set(w, v);
        }
        exec.run();
        let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();

        // Round-trip.
        let _program = garble_eval_roundtrip(
            &mut rng,
            &sys,
            &bits,
            &input_values,
            &output_wires,
            &exec_outputs,
        );

        // CRT reconstruct the evaluator's residues; compare with direct bignum.
        let output_residues: Vec<u64> = exec_outputs.iter().map(|v| v.v).collect();
        let reconstructed = crt_reconstruct(&output_residues, &params.primes);
        for (t, &p_t) in params.primes.iter().enumerate() {
            assert_eq!(reconstructed.mod_u64(p_t), output_residues[t]);
        }
    }
}

#[test]
fn test_garble_program_size_matches_join_width() {
    // Check Theorem 6.1 size bound exactly: program bits = join_width + sum(output bits).
    // Uses a 5-prime primorial for speed; includes p=2 since affine.rs declares
    // coefficient constants NCF, which keeps Z_2 outputs NCF too.
    let primes = [2, 3, 5, 7, 11];
    let params = CrtParams::from_primes(&primes, 12);
    let n = params.n as usize;
    let mut rng = rng();

    let a_residues: Vec<Vec<u64>> =
        params.primes.iter().map(|&pi| vec![1u64 % pi]).collect();
    let b_residues: Vec<Vec<u64>> =
        params.primes.iter().map(|&pi| vec![0u64 % pi]).collect();

    let mut sys = System::new();
    let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
    let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);
    let output_wires: Vec<Wire> = result.outputs.iter().map(|row| row[0]).collect();

    // Compute expected program bits two independent ways and require both
    // to agree with the program — the gate iteration (verifies per-gate
    // accounting) and the System counters (verifies the System's own bookkeeping).
    let mut cf_join_bits: usize = 0;
    let mut ncf_join_bits: usize = 0;
    let mut switch_lsb_bits: usize = 0;
    for g in &sys.gates {
        match g.typ {
            GateType::Join => {
                let m = sys.modulus(g.in0);
                if sys.is_cf(g.in0) {
                    cf_join_bits += LAMBDA * m.trailing_zeros() as usize;
                } else {
                    ncf_join_bits += ((m - 1).ilog2() + 1) as usize;
                }
            }
            GateType::Switch => switch_lsb_bits += 1,
            _ => {}
        }
    }
    let output_bits = expected_output_bits(&sys, &output_wires);
    let expected_total = cf_join_bits + ncf_join_bits + switch_lsb_bits + output_bits;

    // Cross-check against System counters.
    assert_eq!(
        sys.join_complexity_cf + sys.join_complexity_ncf,
        sys.join_complexity,
        "join_complexity_cf + ncf must equal join_complexity"
    );
    assert_eq!(LAMBDA * sys.join_complexity_cf, cf_join_bits);
    assert_eq!(sys.join_complexity_ncf, ncf_join_bits);

    // Run garble (values don't matter for the size check, but we do a full round-trip).
    let x_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();
    let input_values: Vec<Val> = x_bits.iter().map(|&b| Val::new(b, 2)).collect();
    let mut exec = Exec::new(&sys);
    for (&w, &v) in bits.iter().zip(input_values.iter()) {
        exec.set(w, v);
    }
    exec.run();
    let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();
    let program = garble_eval_roundtrip(
        &mut rng,
        &sys,
        &bits,
        &input_values,
        &output_wires,
        &exec_outputs,
    );

    assert_eq!(
        program.total_bits(),
        expected_total,
        "program bits mismatch: got {}, expected {} (cf_join={}, ncf_join={}, switch_lsb={}, output={})",
        program.total_bits(),
        expected_total,
        cf_join_bits,
        ncf_join_bits,
        switch_lsb_bits,
        output_bits
    );
}

#[test]
fn test_hash_count_consistency_and_bulk_rebate() {
    // Verifies the System's hash-count bookkeeping against an independent
    // structural calculation, and checks that the affine pipeline actually
    // forms NCF switch groups (so the bulk-pack rebate is non-trivial).
    //
    // Fixture: S=3 affine maps over a 10-prime primorial. With S>1 every
    // residue's hot_to_ring_bulk registers per-h_p[k] groups of 3 NCF switches
    // sharing one control wire — exactly the case the optimization addresses.
    use crate::system::LAMBDA_BITS;

    let primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let params = CrtParams::from_primes(&primes, 16);
    let n = params.n as usize;
    let s_dim = 3usize;
    let mut rng = rng();

    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|j| (j as u64 + 1) % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| 0u64 % pi).collect())
        .collect();

    let mut sys = System::new();
    let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
    let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);
    let mut output_wires: Vec<Wire> = Vec::new();
    for row in &result.outputs {
        output_wires.extend(row.iter().copied());
    }

    // Compute the expected hash counts structurally:
    //   - CF switch on Z_{2^k}: k λ-bit calls (no bulk benefit).
    //   - Solo NCF switch (not in a group): 1 call.
    //   - Group of S NCF switches in Z_p: ⌈S·lg|R| / λ⌉ calls.
    let mut expected_cf: usize = 0;
    let mut expected_ncf: usize = 0;
    for (gid, g) in sys.gates.iter().enumerate() {
        if !matches!(g.typ, GateType::Switch) {
            continue;
        }
        if sys.is_cf(g.out) {
            let m = sys.modulus(g.out);
            expected_cf += if m <= 1 { 0 } else { m.trailing_zeros() as usize };
        } else if sys.gate_group(gid).is_none() {
            expected_ncf += 1;
        }
    }
    for i in 0..sys.num_switch_groups() {
        let group = sys.switch_group(i);
        let lg_m = ((group.modulus - 1).ilog2() + 1) as usize;
        expected_ncf += (group.members.len() * lg_m).div_ceil(LAMBDA_BITS);
    }

    assert_eq!(
        sys.hash_count_cf, expected_cf,
        "hash_count_cf {} disagrees with structural total {}",
        sys.hash_count_cf, expected_cf,
    );
    assert_eq!(
        sys.hash_count_ncf, expected_ncf,
        "hash_count_ncf {} disagrees with structural total {}",
        sys.hash_count_ncf, expected_ncf,
    );

    // The pipeline must actually exercise bulk packing: at least one group
    // exists, and its rebate is strictly favorable vs. the naive accounting.
    assert!(
        sys.num_switch_groups() > 0,
        "build_s_aff with S>1 should register at least one switch group"
    );
    let mut naive_ncf_for_groups: usize = 0;
    let mut packed_ncf_for_groups: usize = 0;
    for i in 0..sys.num_switch_groups() {
        let group = sys.switch_group(i);
        let lg_m = ((group.modulus - 1).ilog2() + 1) as usize;
        naive_ncf_for_groups += group.members.len();
        packed_ncf_for_groups += (group.members.len() * lg_m).div_ceil(LAMBDA_BITS);
    }
    assert!(
        packed_ncf_for_groups < naive_ncf_for_groups,
        "expected bulk rebate (packed={}, naive={}); pipeline isn't packing",
        packed_ncf_for_groups,
        naive_ncf_for_groups,
    );

    // Run a full round-trip: the bulk-packed garbler+evaluator must still
    // recover the same outputs Exec produces. With H=0 in the placeholder this
    // succeeds iff both sides agree on member slicing (and on which gates are
    // grouped at all).
    let x_bits: Vec<u64> = (0..n).map(|_| rng.random_range(0..2u64)).collect();
    let input_values: Vec<Val> = x_bits.iter().map(|&b| Val::new(b, 2)).collect();
    let mut exec = Exec::new(&sys);
    for (&w, &v) in bits.iter().zip(input_values.iter()) {
        exec.set(w, v);
    }
    exec.run();
    let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();
    let _program = garble_eval_roundtrip(
        &mut rng,
        &sys,
        &bits,
        &input_values,
        &output_wires,
        &exec_outputs,
    );
}

/// Sum of ceil(log2(modulus)) over the (NCF) output wires.
fn expected_output_bits(sys: &System, output_wires: &[Wire]) -> usize {
    output_wires
        .iter()
        .map(|&w| {
            let m = sys.modulus(w);
            if m <= 1 {
                0
            } else {
                ((m - 1).ilog2() + 1) as usize
            }
        })
        .sum()
}

#[test]
fn test_streaming_s_aff_matches_all_at_once() {
    // Run the same fixture through (a) the all-at-once garble→eval and (b) the
    // streaming pipeline; outputs must match bit-for-bit.
    //
    // Picks parameters where bulk-pack is exercised (S>1) and several primes
    // are present, but small enough to keep the all-at-once side fast.
    let primes: [u64; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let params = CrtParams::from_primes(&primes, 16);
    let n = params.n as usize;
    let s_dim = 3usize;
    let mut rng = rng();
    let m: u128 = params.primes.iter().map(|&p| p as u128).product();

    for _ in 0..5 {
        let a_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..m as u64)).collect();
        let b_vals: Vec<u64> = (0..s_dim).map(|_| rng.random_range(0..m as u64)).collect();
        let x: u64 = rng.random_range(0..(1u64 << n));

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

        // (a) All-at-once: build, garble, eval.
        let mut sys = System::new();
        let bit_wires: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
        let result = build_s_aff(&mut sys, &bit_wires, &params, &a_residues, &b_residues);
        let mut output_wires: Vec<Wire> = Vec::with_capacity(params.num_primes * s_dim);
        for row in &result.outputs {
            output_wires.extend(row.iter().copied());
        }
        let input_values: Vec<Val> = input_bits.iter().map(|&b| Val::new(b, 2)).collect();
        let mut exec = Exec::new(&sys);
        for (&w, &v) in bit_wires.iter().zip(input_values.iter()) {
            exec.set(w, v);
        }
        exec.run();
        let exec_outputs: Vec<Val> = output_wires.iter().map(|&w| exec.get(w)).collect();
        let expected_outputs: Vec<Val> = {
            let delta = normalize_delta(rng.random());
            let input_masks: Vec<Label> =
                bit_wires.iter().map(|_| sample_cf_mask(&mut rng, 2)).collect();
            let program = garble(&sys, &bit_wires, &input_masks, &output_wires, delta);
            let got = eval(
                &sys,
                &bit_wires,
                &input_values,
                delta,
                &input_masks,
                &program,
                &output_wires,
            );
            // Sanity: garble→eval recovers the Exec values.
            assert_eq!(got, exec_outputs);
            got
        };
        drop(sys);

        // (b) Streaming.
        let mut pipeline = Pipeline::new(&mut rng);
        let bit_ids: Vec<_> = input_bits
            .iter()
            .map(|&b| pipeline.seed_input_cf_value(&mut rng, 2, b))
            .collect();
        let outputs = build_s_aff_streaming(
            &mut pipeline,
            &bit_ids,
            &input_bits,
            &params,
            &a_residues,
            &b_residues,
        );

        // Streaming returns Vec<Vec<u64>> with shape [num_primes][s_dim]; flatten
        // in the same order build_s_aff lays out (prime-major, then s).
        let mut streaming_flat: Vec<u64> = Vec::with_capacity(params.num_primes * s_dim);
        for row in &outputs {
            streaming_flat.extend(row.iter().copied());
        }

        for (i, (got, expected)) in streaming_flat.iter().zip(expected_outputs.iter()).enumerate() {
            assert_eq!(
                *got, expected.v,
                "stream-vs-all-at-once mismatch at output[{i}]: got {got}, expected {}",
                expected.v
            );
        }
    }
}
