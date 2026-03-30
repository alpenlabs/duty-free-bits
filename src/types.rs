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
