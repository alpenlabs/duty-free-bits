//! Fixed-width 576-bit unsigned integer for CRT reconstruction.
//!
//! The product of the first 80 primes (the 80th primorial) is ~553 bits.
//! U576 = 9 × 64 = 576 bits.

use std::fmt;
use std::ops::{Add, Sub};

/// A 576-bit unsigned integer stored as 9 little-endian u64 limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U576(pub [u64; 9]);

/// The first 80 primes (p_1 = 2, ..., p_80 = 409).
pub const FIRST_80_PRIMES: [u64; 80] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
];

impl U576 {
    /// The zero value.
    pub const ZERO: U576 = U576([0; 9]);
    /// The one value.
    pub const ONE: U576 = U576([1, 0, 0, 0, 0, 0, 0, 0, 0]);

    /// Create from a single u64.
    pub const fn from_u64(v: u64) -> U576 {
        U576([v, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    /// Create from a u128.
    pub const fn from_u128(v: u128) -> U576 {
        U576([v as u64, (v >> 64) as u64, 0, 0, 0, 0, 0, 0, 0])
    }

    /// Convert to u128, returning None if the value exceeds 128 bits.
    pub fn to_u128(&self) -> Option<u128> {
        if self.0[2..].iter().any(|&limb| limb != 0) {
            return None;
        }
        Some(self.0[0] as u128 | ((self.0[1] as u128) << 64))
    }

    /// Multiply by a u64 scalar. Panics on overflow beyond 576 bits.
    pub fn mul_u64(self, rhs: u64) -> U576 {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (i, limb) in out.iter_mut().enumerate() {
            let wide = (self.0[i] as u128) * (rhs as u128) + carry;
            *limb = wide as u64;
            carry = wide >> 64;
        }
        assert!(carry == 0, "U576::mul_u64 overflow");
        U576(out)
    }

    /// Compute self mod m for a small modulus (Horner's method).
    pub fn mod_u64(&self, m: u64) -> u64 {
        let m128 = m as u128;
        let mut result = 0u128;
        for i in (0..9).rev() {
            result = ((result << 64) | self.0[i] as u128) % m128;
        }
        result as u64
    }

    /// Returns true if self > 0.
    pub fn is_nonzero(&self) -> bool {
        self.0.iter().any(|&limb| limb != 0)
    }
}

impl Add for U576 {
    type Output = U576;
    fn add(self, rhs: U576) -> U576 {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (i, limb) in out.iter_mut().enumerate() {
            let sum = (self.0[i] as u128) + (rhs.0[i] as u128) + carry;
            *limb = sum as u64;
            carry = sum >> 64;
        }
        assert!(carry == 0, "U576::add overflow");
        U576(out)
    }
}

impl Sub for U576 {
    type Output = U576;
    fn sub(self, rhs: U576) -> U576 {
        let mut out = [0u64; 9];
        let mut borrow = 0i128;
        for (i, limb) in out.iter_mut().enumerate() {
            let diff = (self.0[i] as i128) - (rhs.0[i] as i128) - borrow;
            if diff < 0 {
                *limb = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                *limb = diff as u64;
                borrow = 0;
            }
        }
        assert!(borrow == 0, "U576::sub underflow");
        U576(out)
    }
}

impl PartialOrd for U576 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U576 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..9).rev() {
            match self.0[i].cmp(&other.0[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl From<u64> for U576 {
    fn from(v: u64) -> Self {
        U576::from_u64(v)
    }
}

impl From<u128> for U576 {
    fn from(v: u128) -> Self {
        U576::from_u128(v)
    }
}

impl fmt::Display for U576 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Find the highest non-zero limb for compact display.
        let top = self.0.iter().rposition(|&l| l != 0).unwrap_or(0);
        write!(f, "0x{:x}", self.0[top])?;
        for i in (0..top).rev() {
            write!(f, "_{:016x}", self.0[i])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_u64_roundtrip() {
        for v in [0u64, 1, 42, u64::MAX] {
            let big = U576::from_u64(v);
            assert_eq!(big.to_u128(), Some(v as u128));
        }
    }

    #[test]
    fn test_from_u128_roundtrip() {
        for v in [0u128, 1, u64::MAX as u128, u64::MAX as u128 + 1, u128::MAX] {
            let big = U576::from_u128(v);
            assert_eq!(big.to_u128(), Some(v));
        }
    }

    #[test]
    fn test_to_u128_overflow() {
        let mut big = U576::ZERO;
        big.0[2] = 1;
        assert_eq!(big.to_u128(), None);
    }

    #[test]
    fn test_add_basic() {
        let a = U576::from_u64(100);
        let b = U576::from_u64(200);
        assert_eq!((a + b).to_u128(), Some(300));
    }

    #[test]
    fn test_add_carry_propagation() {
        let a = U576::from_u64(u64::MAX);
        let b = U576::from_u64(1);
        let sum = a + b;
        assert_eq!(sum.0[0], 0);
        assert_eq!(sum.0[1], 1);
        assert_eq!(sum.to_u128(), Some(1u128 << 64));
    }

    #[test]
    fn test_add_u128_values() {
        let a = U576::from_u128(u128::MAX);
        let b = U576::from_u128(1);
        let sum = a + b;
        assert_eq!(sum.0[0], 0);
        assert_eq!(sum.0[1], 0);
        assert_eq!(sum.0[2], 1);
        assert_eq!(sum.to_u128(), None);
    }

    #[test]
    fn test_mul_u64_basic() {
        let a = U576::from_u64(100);
        assert_eq!(a.mul_u64(200).to_u128(), Some(20_000));
    }

    #[test]
    fn test_mul_u64_carry() {
        let a = U576::from_u64(u64::MAX);
        let result = a.mul_u64(u64::MAX);
        // u64::MAX * u64::MAX = (2^64 - 1)^2 = 2^128 - 2^65 + 1
        let expected = u128::MAX - (1u128 << 65) + 2; // = (2^64-1)^2
        assert_eq!(result.to_u128(), Some(expected));
    }

    #[test]
    fn test_mul_u64_zero() {
        let a = U576::from_u64(12345);
        assert_eq!(a.mul_u64(0), U576::ZERO);
    }

    #[test]
    fn test_mul_u64_one() {
        let a = U576::from_u128(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
        assert_eq!(a.mul_u64(1), a);
    }

    #[test]
    fn test_sub_basic() {
        let a = U576::from_u64(300);
        let b = U576::from_u64(100);
        assert_eq!((a - b).to_u128(), Some(200));
    }

    #[test]
    fn test_sub_borrow() {
        let a = U576::from_u128(1u128 << 64);
        let b = U576::from_u64(1);
        let diff = a - b;
        assert_eq!(diff.to_u128(), Some(u64::MAX as u128));
    }

    #[test]
    fn test_sub_self_is_zero() {
        let a = U576::from_u128(0x0001_0002_0003_0004);
        assert_eq!(a - a, U576::ZERO);
    }

    #[test]
    #[should_panic(expected = "underflow")]
    fn test_sub_underflow_panics() {
        let a = U576::from_u64(1);
        let b = U576::from_u64(2);
        let _ = a - b;
    }

    #[test]
    fn test_ord() {
        let a = U576::from_u64(100);
        let b = U576::from_u64(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, a);

        // Compare across limbs
        let c = U576::from_u128(1u128 << 64);
        assert!(c > U576::from_u64(u64::MAX));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", U576::ZERO), "0x0");
        assert_eq!(format!("{}", U576::from_u64(0xff)), "0xff");
    }

    #[test]
    fn test_primorial_80() {
        // Compute product of first 80 primes.
        let mut result = U576::ONE;
        for &p in &FIRST_80_PRIMES {
            result = result.mul_u64(p);
        }
        // Verify it's > 2^552
        assert!(
            result.0[8] != 0 || result.0[7] != 0,
            "primorial should be large"
        );
        // Verify it fits in 576 bits (no overflow during computation)
        assert!(result.is_nonzero());

        // Verify against known bit-length: the primorial is ~553 bits,
        // meaning the top non-zero limb should be at index 8 (bits 512-575).
        let top = result.0.iter().rposition(|&l| l != 0).unwrap();
        let bit_len = top * 64 + (64 - result.0[top].leading_zeros() as usize);
        assert!(
            (552..=576).contains(&bit_len),
            "primorial bit-length {bit_len} not in expected range [552, 576]"
        );
    }

    #[test]
    fn test_primorial_small_matches_u128() {
        // Product of first 10 primes = 6,469,693,230
        let mut result = U576::ONE;
        for &p in &FIRST_80_PRIMES[..10] {
            result = result.mul_u64(p);
        }
        assert_eq!(result.to_u128(), Some(6_469_693_230));
    }
}
