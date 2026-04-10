//! Core types for the switch system: values over an integer ring, wires, and gates.

use std::fmt;

/// Index into the gate list.
pub type GateId = usize;

/// A wire in the switch system, identified by index.
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    /// Wire index in the system's value/subscription arrays.
    pub wid: usize,
}

/// Gate types supported by the switch system.
#[derive(Clone, Copy, Debug)]
pub enum GateType {
    /// y := x when ctrl=0.
    Switch,
    /// x = y (costs communication).
    Join,
    /// x = y (free when one side is unconstrained).
    SameWire,
    /// out = in0 + in1.
    Add,
    /// out = in0 - in1.
    Sub,
    /// out = param * in0.
    Mul,
    /// out = in0 mod 2^param.
    Mod2k,
    /// out = in0 / 2^param (requires low bits = 0).
    Div2k,
}

/// A gate in the switch system.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    /// The operation this gate performs.
    pub typ: GateType,
    /// Scalar parameter (used by Mul, Mod2k, Div2k).
    pub param: u64,
    /// First input wire.
    pub in0: Wire,
    /// Second input wire (control wire for Switch).
    pub in1: Wire,
    /// Output wire.
    pub out: Wire,
}

/// A value in Z_modulus. Write-once: starts undefined, transitions to defined exactly once.
#[derive(Clone, Copy, Debug)]
pub struct Val {
    /// The concrete value (only meaningful when `defined` is true).
    pub v: u64,
    /// The ring modulus.
    pub modulus: u64,
    /// Whether this value has been assigned.
    pub defined: bool,
}

impl Val {
    /// Create a defined value `v` in Z_modulus.
    pub fn new(v: u64, modulus: u64) -> Val {
        assert!(modulus > 0);
        assert!(v < modulus, "v={} >= modulus={}", v, modulus);
        Val {
            v,
            modulus,
            defined: true,
        }
    }

    /// Create an undefined value in Z_modulus.
    pub fn none(modulus: u64) -> Val {
        Val {
            v: 0,
            modulus,
            defined: false,
        }
    }

    /// Returns true if this value is undefined.
    pub fn is_none(self) -> bool {
        !self.defined
    }

    /// Create a defined value from `v` in Z_{2^bitlen}.
    pub fn from_bits(v: u64, bitlen: u32) -> Val {
        assert!(bitlen < 64);
        let m = 1u64 << bitlen;
        Val::new(v, m)
    }

    /// Create an undefined value in Z_{2^bitlen}.
    pub fn none_bits(bitlen: u32) -> Val {
        assert!(bitlen < 64);
        Val::none(1u64 << bitlen)
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Val) -> bool {
        self.modulus == other.modulus
            && self.defined == other.defined
            && (!self.defined || self.v == other.v)
    }
}
impl Eq for Val {}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.defined {
            write!(f, "{}", self.v)
        } else {
            write!(f, "?")
        }
    }
}

// === Arithmetic on Val ===

/// Switch guard: returns x if ctrl=0, undefined otherwise.
pub fn guard(x: Val, ctrl: Val) -> Val {
    if ctrl.is_none() {
        return Val::none(x.modulus);
    }
    if ctrl.v == 0 { x } else { Val::none(x.modulus) }
}

/// Add two values in the same ring.
pub fn val_add(x: Val, y: Val) -> Val {
    assert_eq!(x.modulus, y.modulus);
    if x.is_none() || y.is_none() {
        return Val::none(x.modulus);
    }
    Val::new((x.v + y.v) % x.modulus, x.modulus)
}

/// Subtract two values in the same ring.
pub fn val_sub(x: Val, y: Val) -> Val {
    assert_eq!(x.modulus, y.modulus);
    if x.is_none() || y.is_none() {
        return Val::none(x.modulus);
    }
    Val::new((x.modulus + x.v - y.v) % x.modulus, x.modulus)
}

/// Scalar multiplication: s * x in x's ring.
pub fn val_mul(s: u64, x: Val) -> Val {
    if x.is_none() {
        return Val::none(x.modulus);
    }
    let p = ((s as u128) * (x.v as u128)) % (x.modulus as u128);
    Val::new(p as u64, x.modulus)
}

/// Reduce x modulo 2^k.
pub fn val_mod2k(x: Val, k: u32) -> Val {
    assert!(k < 64);
    let m = 1u64 << k;
    assert!(m <= x.modulus);
    if x.is_none() {
        return Val::none(m);
    }
    Val::new(x.v % m, m)
}

/// Divide x by 2^k (requires low k bits to be zero).
pub fn val_div2k(x: Val, k: u32) -> Val {
    assert!(k < 64);
    let d = 1u64 << k;
    assert!(x.modulus > d);
    if x.is_none() {
        return Val::none(x.modulus / d);
    }
    assert_eq!(x.v % d, 0, "div2k: {} not divisible by 2^{}", x.v, k);
    Val::new(x.v / d, x.modulus / d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    const SAMPLES: usize = 10;
    const MOD: u64 = 16;

    fn rng() -> impl Rng {
        rand::rng()
    }

    fn rand_val(rng: &mut impl Rng, modulus: u64) -> Val {
        Val::new(rng.random_range(0..modulus), modulus)
    }

    // ==================== Val construction ====================

    #[test]
    fn test_new_and_accessors() {
        let v = Val::new(3, 8);
        assert_eq!(v.v, 3);
        assert_eq!(v.modulus, 8);
        assert!(!v.is_none());
    }

    #[test]
    fn test_none_is_undefined() {
        let v = Val::none(16);
        assert!(v.is_none());
        assert_eq!(v.modulus, 16);
    }

    #[test]
    fn test_from_bits_valid() {
        let v = Val::from_bits(5, 3);
        assert_eq!(v, Val::new(5, 8));
    }

    #[test]
    #[should_panic]
    fn test_from_bits_panics_on_overflow() {
        Val::from_bits(10, 3);
    }

    #[test]
    fn test_none_bits() {
        let v = Val::none_bits(5);
        assert!(v.is_none());
        assert_eq!(v.modulus, 32);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_v_ge_modulus() {
        Val::new(8, 8);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_zero_modulus() {
        Val::new(0, 0);
    }

    #[test]
    fn test_eq_defined() {
        assert_eq!(Val::new(3, 8), Val::new(3, 8));
        assert_ne!(Val::new(3, 8), Val::new(4, 8));
        assert_ne!(Val::new(3, 8), Val::new(3, 16));
    }

    #[test]
    fn test_eq_undefined() {
        assert_eq!(Val::none(8), Val::none(8));
        assert_ne!(Val::none(8), Val::none(16));
        assert_ne!(Val::new(0, 8), Val::none(8));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Val::new(42, 100)), "42");
        assert_eq!(format!("{}", Val::none(100)), "?");
    }

    // ==================== Val arithmetic ====================

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
    fn test_add() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let a = rand_val(&mut rng, MOD);
            let b = rand_val(&mut rng, MOD);
            let result = val_add(a, b);
            assert_eq!(result, Val::new((a.v + b.v) % MOD, MOD));
        }
    }

    #[test]
    fn test_add_undefined_propagates() {
        assert!(val_add(Val::none(8), Val::new(3, 8)).is_none());
        assert!(val_add(Val::new(3, 8), Val::none(8)).is_none());
    }

    #[test]
    fn test_sub() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let a = rand_val(&mut rng, MOD);
            let b = rand_val(&mut rng, MOD);
            let result = val_sub(a, b);
            assert_eq!(result, Val::new((MOD + a.v - b.v) % MOD, MOD));
        }
    }

    #[test]
    fn test_sub_undefined_propagates() {
        assert!(val_sub(Val::none(8), Val::new(3, 8)).is_none());
        assert!(val_sub(Val::new(3, 8), Val::none(8)).is_none());
    }

    #[test]
    fn test_mul() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let s = rng.random_range(0..MOD);
            let x = rand_val(&mut rng, MOD);
            let result = val_mul(s, x);
            assert_eq!(result, Val::new((s * x.v) % MOD, MOD));
        }
    }

    #[test]
    fn test_mul_undefined() {
        assert!(val_mul(3, Val::none(16)).is_none());
    }

    #[test]
    fn test_mod2k() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let x = rand_val(&mut rng, MOD);
            let k = rng.random_range(1..=4);
            let m = 1u64 << k;
            let result = val_mod2k(x, k);
            assert_eq!(result, Val::new(x.v % m, m));
        }
    }

    #[test]
    fn test_mod2k_undefined() {
        assert!(val_mod2k(Val::none(16), 2).is_none());
        assert_eq!(val_mod2k(Val::none(16), 2).modulus, 4);
    }

    #[test]
    fn test_div2k() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let k = rng.random_range(1..4);
            let d = 1u64 << k;
            let quotient = rng.random_range(0..(MOD / d));
            let x = Val::new(quotient * d, MOD);
            let result = val_div2k(x, k);
            assert_eq!(result, Val::new(quotient, MOD / d));
        }
    }

    #[test]
    fn test_div2k_undefined() {
        let r = val_div2k(Val::none(16), 2);
        assert!(r.is_none());
        assert_eq!(r.modulus, 4);
    }

    #[test]
    #[should_panic(expected = "div2k")]
    fn test_div2k_panics_not_divisible() {
        val_div2k(Val::new(5, 16), 2);
    }
}
