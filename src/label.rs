//! Garbled labels.
//!
//! A CF (control-friendly) label on a wire in Z_{2^k} stores LAMBDA coordinates
//! of k bits each, bit-packed.
//!
//! An NCF label is a single ring element,

/// Security parameter: number of coordinates in a control-friendly label.
pub const LAMBDA: usize = 128;

/// A control-friendly label: LAMBDA coordinates in Z_{2^k}, bit-packed.
#[derive(Clone, Debug)]
pub struct CfLabel {
    /// Bit-packed coordinates. Coordinate `i` occupies bits `[i*k .. (i+1)*k)`
    /// of the concatenated bit string, stored LSB-first in `bits[bit/64]`.
    bits: Vec<u64>,
    /// Ring modulus; must be a power of two in `[2, 2^62]`.
    modulus: u64,
}

/// A non-control-friendly label: a single coordinate in Z_modulus.
#[derive(Clone, Copy, Debug)]
pub struct NcfLabel {
    /// The single coordinate.
    pub rep: u64,
    /// Ring modulus.
    pub modulus: u64,
}

/// A garbled label.
#[derive(Clone, Debug)]
pub enum Label {
    /// Control-friendly variant.
    Cf(CfLabel),
    /// Non-control-friendly variant.
    Ncf(NcfLabel),
}

// ---- CfLabel construction / accessors ----

impl CfLabel {
    /// Zero label in Z_{modulus}. `modulus` must be a power of two in `[2, 2^62]`.
    pub fn zero(modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "CF modulus {} is not power of two",
            modulus
        );
        assert!((2..=(1u64 << 62)).contains(&modulus));
        let k = modulus.trailing_zeros() as usize;
        let words = (LAMBDA * k).div_ceil(64);
        CfLabel {
            bits: vec![0u64; words],
            modulus,
        }
    }

    /// Build from an explicit coord list (length LAMBDA).
    pub fn from_coords(coords: &[u64], modulus: u64) -> Self {
        assert_eq!(coords.len(), LAMBDA);
        let mut l = Self::zero(modulus);
        for (i, &c) in coords.iter().enumerate() {
            l.set(i, c);
        }
        l
    }

    /// Unpack coordinates into a Vec<u64>.
    pub fn to_coords(&self) -> Vec<u64> {
        (0..LAMBDA).map(|i| self.get(i)).collect()
    }

    /// Bits per coordinate (= log₂(modulus)).
    pub fn k(&self) -> u32 {
        self.modulus.trailing_zeros()
    }

    /// Ring modulus.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Storage in bits (`LAMBDA * k`).
    pub fn bit_len(&self) -> usize {
        LAMBDA * self.k() as usize
    }

    /// Read coordinate `i`.
    pub fn get(&self, i: usize) -> u64 {
        debug_assert!(i < LAMBDA);
        let k = self.k() as usize;
        if k == 0 {
            return 0;
        }
        let bit_pos = i * k;
        let word = bit_pos / 64;
        let shift = bit_pos % 64;
        let mask = coord_mask(k as u32);
        if shift + k <= 64 {
            (self.bits[word] >> shift) & mask
        } else {
            let lo = self.bits[word] >> shift;
            let hi = self.bits[word + 1] << (64 - shift);
            (lo | hi) & mask
        }
    }

    /// Write coordinate `i`.
    pub fn set(&mut self, i: usize, v: u64) {
        debug_assert!(i < LAMBDA);
        let k = self.k() as usize;
        if k == 0 {
            return;
        }
        let bit_pos = i * k;
        let word = bit_pos / 64;
        let shift = bit_pos % 64;
        let mask = coord_mask(k as u32);
        let v = v & mask;
        self.bits[word] = (self.bits[word] & !(mask << shift)) | (v << shift);
        if shift + k > 64 {
            let high_bits = (shift + k) - 64;
            let high_mask = (1u64 << high_bits) - 1;
            self.bits[word + 1] =
                (self.bits[word + 1] & !high_mask) | ((v >> (64 - shift)) & high_mask);
        }
    }

    /// Raw bit-packed storage (primarily for tests / serialization).
    pub fn raw_bits(&self) -> &[u64] {
        &self.bits
    }

    /// Build a CF label from a raw u64-word storage. The buffer must have
    /// exactly `⌈LAMBDA · k / 64⌉` words.
    pub fn from_raw_bits(mut bits: Vec<u64>, modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "CF modulus {} is not power of two",
            modulus
        );
        let k = modulus.trailing_zeros() as usize;
        let total_bits = LAMBDA * k;
        let words = total_bits.div_ceil(64);
        assert_eq!(bits.len(), words, "raw bits length mismatch");
        let last_used = total_bits % 64;
        if last_used != 0 {
            bits[words - 1] &= (1u64 << last_used) - 1;
        }
        CfLabel { bits, modulus }
    }
}

fn coord_mask(k: u32) -> u64 {
    if k >= 64 { !0u64 } else { (1u64 << k) - 1 }
}

impl PartialEq for CfLabel {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus && (0..LAMBDA).all(|i| self.get(i) == other.get(i))
    }
}
impl Eq for CfLabel {}

impl PartialEq for NcfLabel {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus && self.rep == other.rep
    }
}
impl Eq for NcfLabel {}

impl NcfLabel {
    /// Zero element of Z_modulus.
    pub fn zero(modulus: u64) -> Self {
        NcfLabel { rep: 0, modulus }
    }
}

// ---- Label helpers ----

impl Label {
    /// True iff this label is control-friendly.
    pub fn is_cf(&self) -> bool {
        matches!(self, Label::Cf(_))
    }

    /// Ring modulus.
    pub fn modulus(&self) -> u64 {
        match self {
            Label::Cf(c) => c.modulus,
            Label::Ncf(n) => n.modulus,
        }
    }

    /// CF zero label.
    pub fn zero_cf(modulus: u64) -> Self {
        Label::Cf(CfLabel::zero(modulus))
    }

    /// NCF zero label.
    pub fn zero_ncf(modulus: u64) -> Self {
        Label::Ncf(NcfLabel::zero(modulus))
    }

    /// Zero label, CF iff `is_cf`.
    pub fn zero(is_cf: bool, modulus: u64) -> Self {
        if is_cf {
            Self::zero_cf(modulus)
        } else {
            Self::zero_ncf(modulus)
        }
    }

    /// Borrow as CF; panics if NCF.
    pub fn as_cf(&self) -> &CfLabel {
        if let Label::Cf(c) = self {
            c
        } else {
            panic!("expected CF label")
        }
    }

    /// Borrow as NCF; panics if CF.
    pub fn as_ncf(&self) -> &NcfLabel {
        if let Label::Ncf(n) = self {
            n
        } else {
            panic!("expected NCF label")
        }
    }
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Label::Cf(a), Label::Cf(b)) => a == b,
            (Label::Ncf(a), Label::Ncf(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Label {}

// ---- Arithmetic ----

fn add_mod_pow2(a: u64, b: u64, k: u32) -> u64 {
    let mask = coord_mask(k);
    a.wrapping_add(b) & mask
}

fn sub_mod_pow2(a: u64, b: u64, k: u32) -> u64 {
    let mask = coord_mask(k);
    a.wrapping_sub(b) & mask
}

fn mul_mod_pow2(s: u64, a: u64, k: u32) -> u64 {
    let mask = coord_mask(k);
    s.wrapping_mul(a) & mask
}

/// Coordinate-wise addition.
pub fn add(x: &Label, y: &Label) -> Label {
    match (x, y) {
        (Label::Cf(a), Label::Cf(b)) => {
            assert_eq!(a.modulus, b.modulus, "add: modulus mismatch");
            let k = a.k();
            // Fast path for k=1: XOR the raw packed words directly.
            if k == 1 {
                let mut out = CfLabel::zero(a.modulus);
                for i in 0..a.bits.len() {
                    out.bits[i] = a.bits[i] ^ b.bits[i];
                }
                return Label::Cf(out);
            }
            let mut out = CfLabel::zero(a.modulus);
            for i in 0..LAMBDA {
                out.set(i, add_mod_pow2(a.get(i), b.get(i), k));
            }
            Label::Cf(out)
        }
        (Label::Ncf(a), Label::Ncf(b)) => {
            assert_eq!(a.modulus, b.modulus, "add: modulus mismatch");
            // For our parameter range (p ≤ 409) both operands fit in u32 and
            // their sum in u64; replace the `u128 % p` with one compare-subtract.
            // Invariant: caller maintains a.rep < a.modulus and b.rep < b.modulus,
            // so a.rep + b.rep < 2·modulus and at most one subtraction normalizes.
            let m = a.modulus;
            let s = a.rep + b.rep;
            let rep = if s >= m { s - m } else { s };
            Label::Ncf(NcfLabel { rep, modulus: m })
        }
        _ => panic!("add: CF/NCF mismatch"),
    }
}

/// Coordinate-wise subtraction.
pub fn sub(x: &Label, y: &Label) -> Label {
    match (x, y) {
        (Label::Cf(a), Label::Cf(b)) => {
            assert_eq!(a.modulus, b.modulus, "sub: modulus mismatch");
            let k = a.k();
            if k == 1 {
                let mut out = CfLabel::zero(a.modulus);
                for i in 0..a.bits.len() {
                    out.bits[i] = a.bits[i] ^ b.bits[i];
                }
                return Label::Cf(out);
            }
            let mut out = CfLabel::zero(a.modulus);
            for i in 0..LAMBDA {
                out.set(i, sub_mod_pow2(a.get(i), b.get(i), k));
            }
            Label::Cf(out)
        }
        (Label::Ncf(a), Label::Ncf(b)) => {
            assert_eq!(a.modulus, b.modulus, "sub: modulus mismatch");
            // Same logic as NCF add: branch on the sign instead of `u128 % p`.
            let m = a.modulus;
            let rep = if a.rep >= b.rep {
                a.rep - b.rep
            } else {
                a.rep + m - b.rep
            };
            Label::Ncf(NcfLabel { rep, modulus: m })
        }
        _ => panic!("sub: CF/NCF mismatch"),
    }
}

/// Scalar multiplication by `s` (in each coord's ring). TODO: Check if all this logic is necessary, Claude wrote it.
///
/// Fast paths:
/// * `s = 0` → zero label (no allocation for NCF; trivial vec for CF).
/// * `s mod modulus = 1` → caller's label cloned (no arithmetic).
/// * Z_2 CF: depends only on `s & 1` (single XOR-of-zero or copy).
pub fn scalar_mul(s: u64, x: &Label) -> Label {
    match x {
        Label::Cf(a) => {
            let k = a.k();
            // s = 0 collapses to the zero label regardless of k.
            if s == 0 {
                return Label::Cf(CfLabel::zero(a.modulus));
            }
            // s ≡ 1 mod 2^k → identity.
            let mask_k = coord_mask(k);
            if (s & mask_k) == 1 {
                return Label::Cf(a.clone());
            }
            let mut out = CfLabel::zero(a.modulus);
            if k == 1 {
                // (s & 1) is 1 here (the s=0 path already returned), so just copy.
                out.bits.copy_from_slice(&a.bits);
                return Label::Cf(out);
            }
            for i in 0..LAMBDA {
                out.set(i, mul_mod_pow2(s, a.get(i), k));
            }
            Label::Cf(out)
        }
        Label::Ncf(a) => {
            // For our parameter range (p ≤ 409) the product s·rep fits in u64
            // whenever the caller passes a normalized scalar (s mod p < p) — so
            // we can do a single u64 mul + u64 % p instead of u128. Add fast
            // paths for s ≡ 0 (zero) and s ≡ 1 (identity); these come up a lot
            // (mul-by-coeff with coeff=1 in chunk accumulation, identity truth
            // tables in hot_to_ring_bulk).
            let m = a.modulus;
            // Binding precondition for the single-u64 product below (matches
            // `it_gc::mod_mul`): with m ≤ 2^32 and rep < m, s_mod·rep < 2^64.
            debug_assert!(
                m <= (1u64 << 32),
                "NCF scalar_mul modulus {} exceeds 2^32",
                m
            );
            let s_mod = if s < m { s } else { s % m };
            if s_mod == 0 {
                return Label::Ncf(NcfLabel { rep: 0, modulus: m });
            }
            if s_mod == 1 {
                return Label::Ncf(*a);
            }
            let rep = (s_mod * a.rep) % m;
            Label::Ncf(NcfLabel { rep, modulus: m })
        }
    }
}

/// Reduce each coordinate mod 2^k_out; output modulus becomes 2^k_out.
pub fn mod2k(x: &Label, k_out: u32) -> Label {
    match x {
        Label::Cf(a) => {
            assert!(k_out <= a.k());
            let out_mod = 1u64 << k_out;
            let m_in = coord_mask(a.k());
            let m_out = coord_mask(k_out);
            let mut out = CfLabel::zero(out_mod);
            for i in 0..LAMBDA {
                out.set(i, (a.get(i) & m_in) & m_out);
            }
            Label::Cf(out)
        }
        Label::Ncf(a) => {
            assert!(a.modulus.is_power_of_two());
            assert!(k_out <= a.modulus.trailing_zeros());
            let m_out = 1u64 << k_out;
            Label::Ncf(NcfLabel {
                rep: a.rep & (m_out - 1),
                modulus: m_out,
            })
        }
    }
}

/// Zero low k bits of each coordinate then divide by 2^k.
///
/// Output modulus = input_modulus / 2^k.
pub fn div2k(x: &Label, k: u32) -> Label {
    match x {
        Label::Cf(a) => {
            assert!(a.modulus > (1u64 << k));
            let low_mask = (1u64 << k) - 1;
            let out_mod = a.modulus >> k;
            let mut out = CfLabel::zero(out_mod);
            for i in 0..LAMBDA {
                out.set(i, (a.get(i) & !low_mask) >> k);
            }
            Label::Cf(out)
        }
        Label::Ncf(a) => {
            assert!(a.modulus > (1u64 << k));
            let low_mask = (1u64 << k) - 1;
            Label::Ncf(NcfLabel {
                rep: (a.rep & !low_mask) >> k,
                modulus: a.modulus >> k,
            })
        }
    }
}

/// Build Δ_R for ring R = Z_{modulus}: a CF label whose coord `i` is bit `i`
/// of the global 128-bit Δ.
pub fn delta_r(delta: u128, modulus: u64) -> CfLabel {
    let mut out = CfLabel::zero(modulus);
    for i in 0..LAMBDA {
        let bit = ((delta >> i) & 1) as u64;
        if bit != 0 {
            out.set(i, 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rng() -> impl Rng {
        rand::rng()
    }

    fn rand_cf(rng: &mut impl Rng, modulus: u64) -> CfLabel {
        let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
        CfLabel::from_coords(&coords, modulus)
    }

    fn rand_ncf(rng: &mut impl Rng, modulus: u64) -> NcfLabel {
        NcfLabel {
            rep: rng.random_range(0..modulus),
            modulus,
        }
    }

    // ---- Packing ----

    #[test]
    fn test_z2_label_is_128_bits() {
        let l = CfLabel::zero(2);
        // LAMBDA * 1 bits = 128 bits = 2 u64 words.
        assert_eq!(l.raw_bits().len(), 2);
        assert_eq!(l.bit_len(), 128);
    }

    #[test]
    fn test_z22_label_is_44_words() {
        // k=22, 128*22 = 2816 bits, ceil / 64 = 44 words.
        let l = CfLabel::zero(1 << 22);
        assert_eq!(l.raw_bits().len(), 44);
        assert_eq!(l.bit_len(), 2816);
    }

    #[test]
    fn test_get_set_roundtrip_all_k() {
        for k in 1u32..=22 {
            let m = 1u64 << k;
            let mut l = CfLabel::zero(m);
            let coords: Vec<u64> = (0..LAMBDA as u64).map(|i| i % m).collect();
            for (i, &c) in coords.iter().enumerate() {
                l.set(i, c);
            }
            for (i, &c) in coords.iter().enumerate() {
                assert_eq!(l.get(i), c, "k={}, i={}", k, i);
            }
        }
    }

    #[test]
    fn test_set_does_not_leak_bits_into_neighbors() {
        let mut l = CfLabel::zero(1 << 10);
        l.set(5, 0x3FF); // all ones in coord 5
        for i in 0..LAMBDA {
            if i == 5 {
                assert_eq!(l.get(i), 0x3FF);
            } else {
                assert_eq!(l.get(i), 0, "bleed into coord {}", i);
            }
        }
    }

    #[test]
    fn test_set_overwrites() {
        let mut l = CfLabel::zero(1 << 10);
        l.set(7, 0x3FF);
        l.set(7, 0x123);
        assert_eq!(l.get(7), 0x123);
    }

    // ---- Arithmetic ----

    #[test]
    fn test_add_sub_roundtrip_cf() {
        let mut r = rng();
        for k in [1u32, 4, 10, 22] {
            let m = 1u64 << k;
            let a = rand_cf(&mut r, m);
            let b = rand_cf(&mut r, m);
            let c = add(&Label::Cf(a.clone()), &Label::Cf(b.clone()));
            let d = sub(&c, &Label::Cf(b));
            assert_eq!(d, Label::Cf(a));
        }
    }

    #[test]
    fn test_add_sub_roundtrip_ncf() {
        let mut r = rng();
        for m in [2u64, 3, 7, 409, 97] {
            let a = rand_ncf(&mut r, m);
            let b = rand_ncf(&mut r, m);
            let c = add(&Label::Ncf(a), &Label::Ncf(b));
            let d = sub(&c, &Label::Ncf(b));
            assert_eq!(d, Label::Ncf(a));
        }
    }

    #[test]
    fn test_scalar_mul_distributes_over_add() {
        let mut r = rng();
        let m = 1u64 << 10;
        let a = rand_cf(&mut r, m);
        let b = rand_cf(&mut r, m);
        let s = r.random_range(0..m);
        let lhs = scalar_mul(s, &add(&Label::Cf(a.clone()), &Label::Cf(b.clone())));
        let rhs = add(&scalar_mul(s, &Label::Cf(a)), &scalar_mul(s, &Label::Cf(b)));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_scalar_mul_z2_zero_and_one() {
        let mut r = rng();
        let a = rand_cf(&mut r, 2);
        let z = scalar_mul(0, &Label::Cf(a.clone()));
        assert_eq!(z, Label::Cf(CfLabel::zero(2)));
        let o = scalar_mul(1, &Label::Cf(a.clone()));
        assert_eq!(o, Label::Cf(a));
    }

    #[test]
    fn test_mod2k_reduces_modulus() {
        let mut r = rng();
        let a = rand_cf(&mut r, 1 << 10);
        let b = mod2k(&Label::Cf(a.clone()), 4);
        assert_eq!(b.modulus(), 16);
        for i in 0..LAMBDA {
            assert_eq!(b.as_cf().get(i), a.get(i) & 0xF);
        }
    }

    #[test]
    fn test_div2k_zeros_low_bits_then_shifts() {
        let mut r = rng();
        let a = rand_cf(&mut r, 1 << 10);
        let k = 3u32;
        let b = div2k(&Label::Cf(a.clone()), k);
        let d = 1u64 << k;
        assert_eq!(b.modulus(), (1u64 << 10) / d);
        for i in 0..LAMBDA {
            let expected = (a.get(i) & !(d - 1)) / d;
            assert_eq!(b.as_cf().get(i), expected);
        }
    }

    #[test]
    fn test_delta_r_coords_equal_delta_bits() {
        let delta: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
        let d = delta_r(delta, 1 << 10);
        for i in 0..LAMBDA {
            let bit = ((delta >> i) & 1) as u64;
            assert_eq!(d.get(i), bit);
        }
    }

    #[test]
    #[should_panic(expected = "CF/NCF mismatch")]
    fn test_add_kind_mismatch_panics() {
        let a = Label::zero_cf(16);
        let b = Label::zero_ncf(16);
        let _ = add(&a, &b);
    }
}
