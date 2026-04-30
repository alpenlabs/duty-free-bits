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

/// Format an integer with thousands separators.
fn fmt_n(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

/// Reproduces the simulated communication and hash-count costs reported in
/// the Embryo paper. For each S in {1, 256, 512, ..., 256·20}, builds a
/// fresh n=256, 80-prime affine switch system with S affine components, then
/// reads the counters that have been accumulated during construction:
/// `sys.{join_complexity,hash_count}_{cf,ncf}`. The S=1 build is run through
/// the Exec engine and its output residues are checked against a CRT
/// reconstruction, to verify end-to-end correctness.
///
/// Cost model (per-wire CF flag is set by the circuit builder):
///   - join,  CF payload Z_{2^k}: k · λ bits of communication
///   - join, NCF payload mod m:   ⌈log₂ m⌉ bits of communication
///   - switch,  CF payload k bits: k hash invocations
///   - switch, NCF payload P bits: ⌈P / λ⌉ hash invocations
/// NCF switch packing: when S separate NCF switches share the same OHE-entry
/// control bit (as in the affine output stage), they combine into one bulk
/// switch with payload S · ⌈log₂ m⌉ bits and ⌈S · ⌈log₂ m⌉ / λ⌉ hashes.
///
/// Run with:
///   cargo test --release test_s_aff_80_metrics -- --ignored --nocapture
///
/// Roughly 7 minutes wall time, ~6.7 GB peak RSS at S=5120.
#[test]
#[ignore]
fn test_s_aff_80_metrics() {
    const LAMBDA: usize = 128;
    // Default: full sweep S ∈ {1, 256, 512, ..., 256·20}. For quick
    // iteration on the output format, set FORMATTING_PREVIEW=1 to run only
    // S ∈ {1, 5·256, 7·256} (the Embryo-highlight subset).
    let preview = std::env::var("FORMATTING_PREVIEW").is_ok();
    let report_s: Vec<usize> = if preview {
        vec![1, 256 * 5, 256 * 7]
    } else {
        std::iter::once(1)
            .chain((1..=20).map(|k| 256 * k))
            .collect()
    };

    let params = CrtParams::from_primes(&FIRST_80_PRIMES, 256);
    let n = params.n as usize;
    let m = params.primorial();
    let mut rng = rng();

    let lg_p_sum: usize = params
        .primes
        .iter()
        .map(|&p| (p as u128 - 1).ilog2() as usize + 1)
        .sum();
    let p_sum: u64 = params.primes.iter().sum();

    eprintln!();
    eprintln!("================================================================");
    eprintln!("  Embryo: simulated communication & computation cost");
    eprintln!("================================================================");
    eprintln!(
        "  parameters: n = {} input bits, T = {} primes (p_max = {})",
        params.n,
        params.num_primes,
        params.primes.last().unwrap()
    );
    eprintln!(
        "              ell = {} (working modulus), chunk_size = {}, λ = {}",
        params.ell, params.chunk_size, LAMBDA
    );
    eprintln!(
        "              Σ ⌈log₂ p_i⌉ = {} bits,  Σ p_i = {}",
        lg_p_sum, p_sum
    );
    eprintln!();
    eprintln!("Building S=1 (full Exec, e2e correctness verified)...");
    let s1_counts = {
        let t0 = std::time::Instant::now();
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

        let mut exec = Exec::new(&sys);
        for j in 0..n {
            exec.set(bits[j], Val::new(x_bits[j], 2));
        }
        exec.run();
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
            assert_eq!(output_residues[t], expected, "prime {p_t}");
        }
        let reconstructed = crt_reconstruct(&output_residues, &params.primes);
        for (t, &p_t) in params.primes.iter().enumerate() {
            assert_eq!(reconstructed.mod_u64(p_t), output_residues[t]);
        }
        eprintln!("  S=    1 ok ({:.1}s)", t0.elapsed().as_secs_f64());
        (
            sys.join_complexity_cf,
            sys.join_complexity_ncf,
            sys.hash_count_cf,
            sys.hash_count_ncf,
        )
    };

    // For S>1, build a fresh full System and read the counters.
    // Residue values don't affect counts, so we pass 0s.
    let mut results: Vec<(usize, usize, usize, usize, usize)> =
        Vec::with_capacity(report_s.len());
    for &s in &report_s {
        let (jcf, jncf, hcf, hncf) = if s == 1 {
            s1_counts
        } else {
            let t0 = std::time::Instant::now();
            let a_residues: Vec<Vec<u64>> = params.primes.iter().map(|_| vec![0u64; s]).collect();
            let b_residues: Vec<Vec<u64>> = params.primes.iter().map(|_| vec![0u64; s]).collect();
            let mut sys = System::new();
            let bits: Vec<Wire> = (0..n).map(|_| sys.input(2)).collect();
            let _ = build_s_aff(&mut sys, &bits, &params, &a_residues, &b_residues);
            eprintln!("  S={s:>5} ok ({:.1}s)", t0.elapsed().as_secs_f64());
            (
                sys.join_complexity_cf,
                sys.join_complexity_ncf,
                sys.hash_count_cf,
                sys.hash_count_ncf,
            )
        };
        results.push((s, jcf, jncf, hcf, hncf));
    }

    eprintln!();
    eprintln!("Cost table (one row per independent build of build_s_aff):");
    eprintln!();
    let header = format!(
        "  {:>5}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "S", "Comm KiB", "Comm bits", "CF hashes", "NCF hashes", "Total hashes"
    );
    eprintln!("{header}");
    eprintln!("  {}", "-".repeat(header.len() - 2));
    for (s, jcf, jncf, hcf, hncf) in &results {
        let comm_bits = jcf * LAMBDA + jncf;
        let kib = comm_bits as f64 / 8.0 / 1024.0;
        let hashes = hcf + hncf;
        eprintln!(
            "  {s:>5}  {kib:>10.2}  {:>14}  {:>14}  {:>14}  {:>14}",
            fmt_n(comm_bits),
            fmt_n(*hcf),
            fmt_n(*hncf),
            fmt_n(hashes),
        );
    }

    eprintln!();
    eprintln!("Highlights (paper scenarios at n=256):");
    let lookup = |s: usize| {
        results
            .iter()
            .find(|(rs, ..)| *rs == s)
            .copied()
            .map(|(_, jcf, jncf, hcf, hncf)| (jcf * LAMBDA + jncf, hcf + hncf))
    };
    let print_row = |label: &str, s_label: &str, comm_bits: usize, hashes: usize| {
        let kib = comm_bits as f64 / 8.0 / 1024.0;
        let hashes_m = hashes as f64 / 1.0e6;
        eprintln!(
            "  {label:<36}  {s_label:<18}  comm = {kib:>7.2} KiB    hashes = {hashes_m:>5.2} M"
        );
    };
    if let Some((c, h)) = lookup(1) {
        print_row("One-time CRT conversion", "S = 1", c, h);
    }
    // Embryo scalar mul: D = 12 components split across x (5) and y (7),
    // each with its own CRT conversion + IT-GS bulk. Total = sum of two
    // independent builds at S=5·256 and S=7·256.
    if let (Some((c_x, h_x)), Some((c_y, h_y))) = (lookup(5 * 256), lookup(7 * 256)) {
        print_row(
            "Embryo scalar mul (5·256 + 7·256)",
            "S = 1280 + 1792",
            c_x + c_y,
            h_x + h_y,
        );
    }
}
