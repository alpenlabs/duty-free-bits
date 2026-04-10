//! Integration tests for the switch system, execution engine, and affine maps.
//!
//! Unit tests for individual modules live in their respective `#[cfg(test)]` blocks:
//! - `types.rs`: Val arithmetic
//! - `components/ohe.rs`: one-hot encoding
//! - `components/convert.rs`: bin_to_word, word_to_hot, word_to_ring, hot_to_ring
//! - `components/crt.rs`: CRT parameters and reconstruction
//! - `components/bigint.rs`: U576 arithmetic

use crate::components::affine::build_s_aff;
use crate::components::bigint::{FIRST_80_PRIMES, U576};
use crate::components::crt::{CrtParams, crt_reconstruct};
use crate::exec::Exec;
use crate::system::System;
use crate::types::*;

use rand::Rng;

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
    // Full 80-prime CRT pipeline with target parameters: n=256, ell=23.
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
    // Run with: N=12 /usr/bin/time -l cargo test --release test_s_aff_scaling -- --ignored --nocapture
    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .expect("N must be a u32");

    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    eprintln!(
        "n={n}, ell={}, chunk_size={}, num_chunks={}",
        params.ell, params.chunk_size, params.num_chunks
    );
    eprintln!("table_size = 2^{} = {}", params.ell, 1u64 << params.ell);

    let mut rng = rng();
    let max_x = 1u64 << n;
    let a: u64 = rng.random_range(0..1u64 << 48);
    let b: u64 = rng.random_range(0..1u64 << 48);
    let x: u64 = rng.random_range(0..max_x);

    let a_residues: Vec<Vec<u64>> = params.primes.iter().map(|&pi| vec![a % pi]).collect();
    let b_residues: Vec<Vec<u64>> = params.primes.iter().map(|&pi| vec![b % pi]).collect();

    let mut sys = System::new();
    let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();

    eprintln!("building switch system...");
    let result = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);
    eprintln!(
        "system built: {} wires, {} gates",
        sys.num_wires(),
        sys.num_gates()
    );

    eprintln!("executing...");
    let mut exec = Exec::new(&sys);
    for j in 0..n {
        exec.set(bits[j as usize], Val::new((x >> j) & 1, 2));
    }
    exec.run();

    let residues: Vec<u64> = result
        .outputs
        .iter()
        .map(|prime_outs| exec.get(prime_outs[0]).v)
        .collect();
    let reconstructed = crt_reconstruct(&residues, &params.primes);
    let expected = a * x + b;
    assert_eq!(reconstructed, U576::from_u64(expected));
    eprintln!("ok: a*x+b = {expected}");
}
