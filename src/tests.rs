use crate::components::convert::{
    arith_ohe_to_word, bin_to_word, hot_to_ring, word_to_hot, word_to_ring,
};
use crate::components::ohe::{ohe, ohe_scale};
use crate::exec::Exec;
use crate::system::System;
use crate::types::*;

use rand::Rng;

const SAMPLES: usize = 10;
const MOD: u64 = 16;

fn rng() -> impl Rng {
    rand::rng()
}

fn rand_val(rng: &mut impl Rng, modulus: u64) -> Val {
    Val::new(rng.random_range(0..modulus), modulus)
}

// ==================== Val tests ====================

#[test]
fn val_new_and_accessors() {
    let v = Val::new(3, 8);
    assert_eq!(v.v, 3);
    assert_eq!(v.modulus, 8);
    assert!(!v.is_none());
}

#[test]
fn val_none_is_undefined() {
    let v = Val::none(16);
    assert!(v.is_none());
    assert_eq!(v.modulus, 16);
}

#[test]
fn val_from_bits_valid() {
    let v = Val::from_bits(5, 3);
    assert_eq!(v, Val::new(5, 8));
}

#[test]
#[should_panic]
fn val_from_bits_panics_on_overflow() {
    Val::from_bits(10, 3);
}

#[test]
fn val_none_bits() {
    let v = Val::none_bits(5);
    assert!(v.is_none());
    assert_eq!(v.modulus, 32);
}

#[test]
#[should_panic]
fn val_new_panics_on_v_ge_modulus() {
    Val::new(8, 8);
}

#[test]
#[should_panic]
fn val_new_panics_on_zero_modulus() {
    Val::new(0, 0);
}

#[test]
fn val_eq_defined() {
    assert_eq!(Val::new(3, 8), Val::new(3, 8));
    assert_ne!(Val::new(3, 8), Val::new(4, 8));
    assert_ne!(Val::new(3, 8), Val::new(3, 16));
}

#[test]
fn val_eq_undefined() {
    // Two undefined vals with same modulus are equal
    assert_eq!(Val::none(8), Val::none(8));
    // Different modulus → not equal
    assert_ne!(Val::none(8), Val::none(16));
    // Defined vs undefined → not equal
    assert_ne!(Val::new(0, 8), Val::none(8));
}

#[test]
fn val_display() {
    assert_eq!(format!("{}", Val::new(42, 100)), "42");
    assert_eq!(format!("{}", Val::none(100)), "?");
}

// ==================== Val arithmetic tests ====================

#[test]
fn test_guard_ctrl_zero_passes() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let x = rand_val(&mut rng, MOD);
        let ctrl = Val::new(0, 2);
        assert_eq!(guard(x, ctrl), x);
    }
}

#[test]
fn test_guard_ctrl_one_blocks() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let x = rand_val(&mut rng, MOD);
        let ctrl = Val::new(1, 2);
        assert!(guard(x, ctrl).is_none());
    }
}

#[test]
fn test_guard_ctrl_undefined() {
    let x = Val::new(5, 8);
    let ctrl = Val::none(2);
    assert!(guard(x, ctrl).is_none());
}

#[test]
fn test_val_add() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rand_val(&mut rng, MOD);
        let b = rand_val(&mut rng, MOD);
        let result = val_add(a, b);
        assert_eq!(result, Val::new((a.v + b.v) % MOD, MOD));
    }
}

#[test]
fn test_val_add_undefined_propagates() {
    assert!(val_add(Val::none(8), Val::new(3, 8)).is_none());
    assert!(val_add(Val::new(3, 8), Val::none(8)).is_none());
}

#[test]
fn test_val_sub() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let a = rand_val(&mut rng, MOD);
        let b = rand_val(&mut rng, MOD);
        let result = val_sub(a, b);
        assert_eq!(result, Val::new((MOD + a.v - b.v) % MOD, MOD));
    }
}

#[test]
fn test_val_sub_undefined_propagates() {
    assert!(val_sub(Val::none(8), Val::new(3, 8)).is_none());
    assert!(val_sub(Val::new(3, 8), Val::none(8)).is_none());
}

#[test]
fn test_val_mul() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let s = rng.random_range(0..MOD);
        let x = rand_val(&mut rng, MOD);
        let result = val_mul(s, x);
        assert_eq!(result, Val::new((s * x.v) % MOD, MOD));
    }
}

#[test]
fn test_val_mul_undefined() {
    assert!(val_mul(3, Val::none(16)).is_none());
}

#[test]
fn test_val_mod2k() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let x = rand_val(&mut rng, MOD);
        let k = rng.random_range(1..=4); // 1..=log2(16)
        let m = 1u64 << k;
        let result = val_mod2k(x, k);
        assert_eq!(result, Val::new(x.v % m, m));
    }
}

#[test]
fn test_val_mod2k_undefined() {
    assert!(val_mod2k(Val::none(16), 2).is_none());
    assert_eq!(val_mod2k(Val::none(16), 2).modulus, 4);
}

#[test]
fn test_val_div2k() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let k = rng.random_range(1..4); // 1..3 so output modulus >= 2
        let d = 1u64 << k;
        // Pick a value divisible by 2^k
        let quotient = rng.random_range(0..(MOD / d));
        let x = Val::new(quotient * d, MOD);
        let result = val_div2k(x, k);
        assert_eq!(result, Val::new(quotient, MOD / d));
    }
}

#[test]
fn test_val_div2k_undefined() {
    let r = val_div2k(Val::none(16), 2);
    assert!(r.is_none());
    assert_eq!(r.modulus, 4);
}

#[test]
#[should_panic(expected = "div2k")]
fn test_val_div2k_panics_not_divisible() {
    val_div2k(Val::new(5, 16), 2);
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

// ==================== OHE tests ====================

#[test]
fn test_ohe_1bit() {
    // 1-bit OHE: input 0 → [1,0], input 1 → [0,1]
    for b in 0..2u64 {
        let mut sys = System::new();
        let x = sys.input(2);
        let h = ohe(&mut sys, &[x]);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::new(b, 2));
        exec.run();

        assert_eq!(h.len(), 2);
        for (i, &w) in h.iter().enumerate() {
            let expected = if i as u64 == b { 1 } else { 0 };
            assert_eq!(exec.get(w), Val::new(expected, 2), "ohe_1bit({b})[{i}]");
        }
    }
}

#[test]
fn test_ohe_2bit() {
    for input in 0..4u64 {
        let mut sys = System::new();
        let b0 = sys.input(2);
        let b1 = sys.input(2);
        let h = ohe(&mut sys, &[b0, b1]);

        let mut exec = Exec::new(&sys);
        exec.set(b0, Val::new(input & 1, 2));
        exec.set(b1, Val::new((input >> 1) & 1, 2));
        exec.run();

        assert_eq!(h.len(), 4);
        for (i, &w) in h.iter().enumerate() {
            let expected = if i as u64 == input { 1 } else { 0 };
            assert_eq!(exec.get(w), Val::new(expected, 2), "ohe_2bit({input})[{i}]");
        }
    }
}

#[test]
fn test_ohe_scale_basic() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let s_val = rng.random_range(0..MOD);
        let hot_idx = rng.random_range(0..4usize);

        let mut sys = System::new();
        let h: Vec<Wire> = (0..4).map(|_| sys.input(2)).collect();
        let s = sys.input(MOD);
        let out = ohe_scale(&mut sys, &h, s);

        let mut exec = Exec::new(&sys);
        for (i, &w) in h.iter().enumerate() {
            exec.set(w, Val::new(if i == hot_idx { 1 } else { 0 }, 2));
        }
        exec.set(s, Val::new(s_val, MOD));
        exec.run();

        for (i, &w) in out.iter().enumerate() {
            let expected = if i == hot_idx { s_val } else { 0 };
            assert_eq!(exec.get(w), Val::new(expected, MOD));
        }
    }
}

// ==================== Convert tests ====================

#[test]
fn test_arith_ohe_to_word() {
    let mut rng = rng();
    for _ in 0..SAMPLES {
        let hot_idx = rng.random_range(0..4usize);
        let mut sys = System::new();
        let h: Vec<Wire> = (0..4).map(|_| sys.input(8)).collect();
        let out = arith_ohe_to_word(&mut sys, &h);

        let mut exec = Exec::new(&sys);
        for (i, &w) in h.iter().enumerate() {
            let v = if i == hot_idx { 1 } else { 0 };
            exec.set(w, Val::new(v, 8));
        }
        exec.run();

        assert_eq!(exec.get(out), Val::new(hot_idx as u64, 8));
    }
}

#[test]
fn word_to_hot_4bit() {
    let mut sys = System::new();
    let x = sys.input_bits(4);
    let hot = word_to_hot(&mut sys, x);

    let mut exec = Exec::new(&sys);
    exec.set(x, Val::from_bits(11, 4));
    exec.run();

    for (i, &w) in hot.iter().enumerate() {
        let v = exec.get(w);
        if i == 11 {
            assert_eq!(v, Val::new(1, 16), "hot[{i}] should be 1");
        } else {
            assert_eq!(v, Val::new(0, 16), "hot[{i}] should be 0");
        }
    }
}

#[test]
fn word_to_hot_all_values_2bit() {
    for input in 0..4u64 {
        let mut sys = System::new();
        let x = sys.input_bits(2);
        let hot = word_to_hot(&mut sys, x);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::from_bits(input, 2));
        exec.run();

        for (i, &w) in hot.iter().enumerate() {
            let expected = if i as u64 == input { 1 } else { 0 };
            assert_eq!(
                exec.get(w),
                Val::new(expected, 4),
                "word_to_hot({input})[{i}]"
            );
        }
    }
}

#[test]
fn test_bin_to_word() {
    // bits [1, 0, 1] (LSB first) = 0b101 = 5, into Z_{2^4}
    let mut sys = System::new();
    let b0 = sys.input(2);
    let b1 = sys.input(2);
    let b2 = sys.input(2);
    let out = bin_to_word(&mut sys, &[b0, b1, b2], 4);

    let mut exec = Exec::new(&sys);
    exec.set(b0, Val::new(1, 2));
    exec.set(b1, Val::new(0, 2));
    exec.set(b2, Val::new(1, 2));
    exec.run();

    assert_eq!(exec.get(out), Val::new(5, 16));
}

#[test]
fn test_bin_to_word_all_values_2bit() {
    for input in 0..4u64 {
        let mut sys = System::new();
        let b0 = sys.input(2);
        let b1 = sys.input(2);
        let out = bin_to_word(&mut sys, &[b0, b1], 3);

        let mut exec = Exec::new(&sys);
        exec.set(b0, Val::new(input & 1, 2));
        exec.set(b1, Val::new((input >> 1) & 1, 2));
        exec.run();

        assert_eq!(exec.get(out), Val::new(input, 8), "bin_to_word({input})");
    }
}

#[test]
fn test_hot_to_ring() {
    // Identity function via truth table on 2-bit OHE
    // g(x) = x, a=1, b=0, r_mod = 8 → computes 1·x + 0 = x
    let truth_table = vec![0, 1, 2, 3];
    let r_mod = 8u64;

    for input in 0..4u64 {
        let mut sys = System::new();
        let b0 = sys.input(2);
        let b1 = sys.input(2);
        let h = ohe(&mut sys, &[b0, b1]);
        let a = sys.constant(1, r_mod);
        let b = sys.constant(0, r_mod);
        let out = hot_to_ring(&mut sys, &h, &truth_table, a, b);

        let mut exec = Exec::new(&sys);
        exec.set(b0, Val::new(input & 1, 2));
        exec.set(b1, Val::new((input >> 1) & 1, 2));
        exec.run();

        assert_eq!(
            exec.get(out),
            Val::new(input, r_mod),
            "hot_to_ring({input})"
        );
    }
}

#[test]
fn word_to_ring_square() {
    // g(x) = x^2 mod 32, a=1, b=0 → computes x^2 mod 32
    let truth_table: Vec<u64> = (0u64..8).map(|x| (x * x) % 32).collect();
    let r_mod = 32u64;

    for input in 0u64..8 {
        let mut sys = System::new();
        let x = sys.input_bits(3);
        let a = sys.constant(1, r_mod);
        let b = sys.constant(0, r_mod);
        let out = word_to_ring(&mut sys, x, &truth_table, a, b);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::from_bits(input, 3));
        exec.run();

        let expected = (input * input) % r_mod;
        assert_eq!(
            exec.get(out),
            Val::new(expected, r_mod),
            "g({input}) should be {expected}"
        );
    }
}

#[test]
fn word_to_ring_identity() {
    // g(x) = x, a=1, b=0, from Z_4 → Z_8
    let truth_table = vec![0, 1, 2, 3];
    let r_mod = 8u64;

    for input in 0..4u64 {
        let mut sys = System::new();
        let x = sys.input_bits(2);
        let a = sys.constant(1, r_mod);
        let b = sys.constant(0, r_mod);
        let out = word_to_ring(&mut sys, x, &truth_table, a, b);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::from_bits(input, 2));
        exec.run();

        assert_eq!(exec.get(out), Val::new(input, r_mod), "id({input})");
    }
}

// ==================== Exec isolation ====================

#[test]
fn exec_does_not_mutate_system() {
    let mut sys = System::new();
    let x = sys.input(8);
    let y = sys.input(8);
    sys.add(x, y);

    // First run
    let mut exec1 = Exec::new(&sys);
    exec1.set(x, Val::new(3, 8));
    exec1.set(y, Val::new(5, 8));
    exec1.run();

    // Second run with different inputs on the same system
    let mut exec2 = Exec::new(&sys);
    exec2.set(x, Val::new(1, 8));
    exec2.set(y, Val::new(2, 8));
    exec2.run();

    // System's own values should still be undefined for inputs
    assert!(sys.values[x.wid].is_none());
    assert!(sys.values[y.wid].is_none());
}
