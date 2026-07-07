//! Garbled labels.
//!
//! A **label** is one party's share of a wire. A wire value `v` is held as a
//! sharing of `v·Δ` for the global secret offset `Δ`: `(eval share) −
//! (garbler share) = v·Δ`. This type is that one share — LAMBDA coordinates of
//! k bits each over a ring Z_{2^k}. The *wire format* is bit-packed; in memory,
//! boolean labels (k = 1) keep that packed form inline while wider arithmetic
//! labels store one coordinate per u32 lane (see [`Label`]).

/// Security parameter: number of coordinates in a label.
pub const LAMBDA: usize = 128;

/// Words in a bit-packed boolean label. Inline (no heap): boolean labels are the
/// most numerous objects, so `Repr::Bits` stores a fixed array.
const BITS_WORDS: usize = LAMBDA.div_ceil(64);
// The inline representation assumes LAMBDA = 128: exactly 2 words and no
// partial tail word (so packing/unpacking never masks a final word).
const _: () = assert!(LAMBDA == 128 && BITS_WORDS == 2);

/// A garbled label: one party's share of a wire, LAMBDA coordinates in Z_{2^k}.
///
/// The wire value lives in the *difference* of the two shares, not in either
/// alone: `(eval share) − (garbler share) = v·Δ`. A **boolean label** (k = 1)
/// is XOR shares over Z_2 (`v ∈ {0,1}`); an **arithmetic label** (k > 1) is
/// additive shares over Z_{2^k}.
///
/// The *wire format* (communication, hashing) is always the bit-packed string
/// where coordinate `i` occupies bits `[i*k .. (i+1)*k)` LSB-first — see
/// [`Label::from_raw_bits`] / [`Label::to_packed_words`]. In memory, k = 1
/// keeps that packed form (2 words, XOR-friendly), while k > 1 stores one
/// coordinate per u32 lane so the λ-wide ring ops are plain vectorizable loops
/// instead of per-coordinate bit surgery.
#[derive(Clone, Debug)]
pub struct Label {
    repr: Repr,
    /// Ring modulus; must be a power of two in `[2, 2^32]`.
    modulus: u64,
}

/// In-memory coordinate storage (see [`Label`]).
#[derive(Clone, Debug)]
enum Repr {
    /// k == 1: λ bits packed into ⌈λ/64⌉ u64 words (the wire format itself),
    /// stored inline.
    Bits([u64; BITS_WORDS]),
    /// 2 ≤ k ≤ 32: one coordinate per u32 lane, length λ.
    Lanes(Vec<u32>),
}

// ---- construction / accessors ----

impl Label {
    /// Zero label in Z_{modulus}. `modulus` must be a power of two in `[2, 2^32]`.
    pub fn zero(modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "modulus {} is not power of two",
            modulus
        );
        assert!((2..=(1u64 << 32)).contains(&modulus));
        let k = modulus.trailing_zeros() as usize;
        let repr = if k == 1 {
            Repr::Bits([0u64; BITS_WORDS])
        } else {
            Repr::Lanes(vec![0u32; LAMBDA])
        };
        Label { repr, modulus }
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

    /// Unpack coordinates into a `Vec<u64>`.
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
        match &self.repr {
            Repr::Bits(bits) => (bits[i / 64] >> (i % 64)) & 1,
            Repr::Lanes(lanes) => lanes[i] as u64,
        }
    }

    /// Write coordinate `i`.
    pub fn set(&mut self, i: usize, v: u64) {
        debug_assert!(i < LAMBDA);
        let mask = coord_mask(self.modulus.trailing_zeros());
        let v = v & mask;
        match &mut self.repr {
            Repr::Bits(bits) => {
                bits[i / 64] = (bits[i / 64] & !(1u64 << (i % 64))) | (v << (i % 64));
            }
            Repr::Lanes(lanes) => lanes[i] = v as u32,
        }
    }

    /// The u32 lane array of an arithmetic (k > 1) label (in-memory form; panics
    /// on a boolean label).
    pub(crate) fn lanes(&self) -> &[u32] {
        match &self.repr {
            Repr::Lanes(l) => l,
            Repr::Bits(_) => panic!("lanes: Z_2 labels are stored bit-packed"),
        }
    }

    /// Build an arithmetic (k > 1) label directly from its lane array.
    pub(crate) fn from_lanes(lanes: Vec<u32>, modulus: u64) -> Label {
        assert!(
            modulus.is_power_of_two() && modulus > 2 && modulus <= (1u64 << 32),
            "from_lanes: modulus {modulus} out of the lanes range"
        );
        assert_eq!(lanes.len(), LAMBDA);
        debug_assert!({
            let mask = coord_mask(modulus.trailing_zeros()) as u32;
            lanes.iter().all(|&l| l & !mask == 0)
        });
        Label {
            repr: Repr::Lanes(lanes),
            modulus,
        }
    }

    /// Raw bit-packed words of a boolean label (the wire format). Boolean only —
    /// arithmetic labels store lanes; serialize those with
    /// [`to_packed_words`](Label::to_packed_words).
    pub fn raw_bits(&self) -> &[u64] {
        match &self.repr {
            Repr::Bits(bits) => bits,
            Repr::Lanes(_) => panic!("raw_bits: only Z_2 labels are stored bit-packed"),
        }
    }

    /// Serialize to the canonical bit-packed wire format: coordinate `i` at
    /// bits `[i*k .. (i+1)*k)` LSB-first, `⌈λ·k/64⌉` words.
    pub fn to_packed_words(&self) -> Vec<u64> {
        match &self.repr {
            Repr::Bits(bits) => bits.to_vec(),
            Repr::Lanes(lanes) => {
                let k = self.k() as usize;
                let mask = coord_mask(k as u32);
                let words = (LAMBDA * k).div_ceil(64);
                let mut bits = vec![0u64; words];
                let (mut word, mut shift) = (0usize, 0usize);
                for &c in lanes {
                    let v = (c as u64) & mask;
                    bits[word] |= v << shift;
                    if shift + k > 64 {
                        bits[word + 1] |= v >> (64 - shift);
                    }
                    shift += k;
                    if shift >= 64 {
                        shift -= 64;
                        word += 1;
                    }
                }
                bits
            }
        }
    }

    /// Build a label from the bit-packed wire format. The buffer must have
    /// exactly `⌈LAMBDA · k / 64⌉` words.
    pub fn from_raw_bits(bits: Vec<u64>, modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "modulus {} is not power of two",
            modulus
        );
        let k = modulus.trailing_zeros() as usize;
        let words = (LAMBDA * k).div_ceil(64);
        assert_eq!(bits.len(), words, "raw bits length mismatch");
        if k == 1 {
            // LAMBDA = 128 (const-asserted): both words fully used, no tail mask.
            return Label {
                repr: Repr::Bits([bits[0], bits[1]]),
                modulus,
            };
        }
        assert!(k <= 32, "modulus 2^{k} exceeds the u32 lane width");
        let mut buf = [0u64; MAX_PACKED_WORDS];
        buf[..words].copy_from_slice(&bits);
        Label {
            repr: Repr::Lanes(unpack_lanes_words(k, &buf)),
            modulus,
        }
    }
}

/// Packed words of the widest supported label (k = 32), plus one pad word so
/// the branchless two-word window reads in [`unpack_lanes_words`] never index
/// out of bounds.
const MAX_PACKED_WORDS: usize = (LAMBDA * 32) / 64 + 1;

/// Unpack LAMBDA packed `k`-bit coordinates (coordinate `i` at bits
/// `[i*k .. (i+1)*k)` LSB-first) into u32 lanes. `padded` is the packed words
/// extended with at least one zero pad word, so every lane is read with one
/// branchless two-word window (no per-lane boundary case, no serial cursor).
fn unpack_lanes_words(k: usize, padded: &[u64; MAX_PACKED_WORDS]) -> Vec<u32> {
    debug_assert!((2..=32).contains(&k));
    let mask = coord_mask(k as u32);
    let mut lanes = vec![0u32; LAMBDA];
    for (i, lane) in lanes.iter_mut().enumerate() {
        let bit = i * k;
        let window = (padded[bit >> 6] as u128) | ((padded[(bit >> 6) + 1] as u128) << 64);
        *lane = ((window >> (bit & 63)) as u64 & mask) as u32;
    }
    lanes
}

fn coord_mask(k: u32) -> u64 {
    if k >= 64 { !0u64 } else { (1u64 << k) - 1 }
}

/// Word-wise XOR of two packed-bit boolean labels (Z_2 add/sub).
fn xor_words(a: &[u64; BITS_WORDS], b: &[u64; BITS_WORDS]) -> [u64; BITS_WORDS] {
    let mut out = [0u64; BITS_WORDS];
    for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b)) {
        *o = x ^ y;
    }
    out
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus && (0..LAMBDA).all(|i| self.get(i) == other.get(i))
    }
}
impl Eq for Label {}

// ---- Arithmetic (all labels here are boolean or arithmetic labels over a power-of-two ring) ----

/// Coordinate-wise addition.
pub fn add(a: &Label, b: &Label) -> Label {
    assert_eq!(a.modulus, b.modulus, "add: modulus mismatch");
    let repr = match (&a.repr, &b.repr) {
        // k = 1: addition mod 2 is XOR on the packed words.
        (Repr::Bits(wa), Repr::Bits(wb)) => Repr::Bits(xor_words(wa, wb)),
        (Repr::Lanes(la), Repr::Lanes(lb)) => {
            let mask = coord_mask(a.k()) as u32;
            Repr::Lanes(
                la.iter()
                    .zip(lb)
                    .map(|(&x, &y)| x.wrapping_add(y) & mask)
                    .collect(),
            )
        }
        _ => unreachable!("repr is determined by the (equal) modulus"),
    };
    Label {
        repr,
        modulus: a.modulus,
    }
}

/// Scalar multiplication by `s` (in each coordinate's ring).
///
/// Fast paths: `s = 0` → zero label; `s ≡ 1 mod 2^k` → clone (no arithmetic);
/// a boolean label depends only on `s & 1` (single XOR-of-zero or copy).
pub fn scalar_mul(s: u64, a: &Label) -> Label {
    let k = a.k();
    // s = 0 collapses to the zero label regardless of k.
    if s == 0 {
        return Label::zero(a.modulus);
    }
    // s ≡ 1 mod 2^k → identity. For k = 1 this covers every nonzero s
    // (the s = 0 path already returned), so only Lanes reach the loop.
    let mask = coord_mask(k);
    if (s & mask) == 1 {
        return a.clone();
    }
    let Repr::Lanes(la) = &a.repr else {
        unreachable!("k = 1 scalars are 0 or 1 and both returned above");
    };
    let s32 = (s & mask) as u32;
    let lanes = la
        .iter()
        .map(|&x| s32.wrapping_mul(x) & mask as u32)
        .collect();
    Label {
        repr: Repr::Lanes(lanes),
        modulus: a.modulus,
    }
}

/// Build Δ_R for ring R = Z_{modulus}: a label whose coord `i` is bit `i`
/// of the global 128-bit Δ.
pub fn delta_r(delta: u128, modulus: u64) -> Label {
    let mut out = Label::zero(modulus);
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
    use rand::{Rng, RngExt};

    fn rng() -> impl Rng {
        rand::rng()
    }

    fn rand_label(rng: &mut impl Rng, modulus: u64) -> Label {
        let coords: Vec<u64> = (0..LAMBDA).map(|_| rng.random_range(0..modulus)).collect();
        Label::from_coords(&coords, modulus)
    }

    // ---- Packing ----

    #[test]
    fn test_z2_label_is_128_bits() {
        let l = Label::zero(2);
        // LAMBDA * 1 bits = 128 bits = 2 u64 words.
        assert_eq!(l.raw_bits().len(), 2);
        assert_eq!(l.bit_len(), 128);
    }

    #[test]
    fn test_z22_label_is_44_packed_words() {
        // k=22, 128*22 = 2816 bits, ceil / 64 = 44 words on the wire.
        let l = Label::zero(1 << 22);
        assert_eq!(l.to_packed_words().len(), 44);
        assert_eq!(l.bit_len(), 2816);
    }

    #[test]
    fn test_packed_words_roundtrip() {
        // from_raw_bits ∘ to_packed_words must be the identity for every k:
        // the packed form is the wire format, the lane form is in-memory only.
        let mut r = rng();
        for k in 1u32..=32 {
            let m = 1u64 << k;
            let l = rand_label(&mut r, m);
            let packed = l.to_packed_words();
            assert_eq!(packed.len(), (LAMBDA * k as usize).div_ceil(64));
            let back = Label::from_raw_bits(packed, m);
            assert_eq!(back, l, "k={k}");
        }
    }

    #[test]
    fn test_get_set_roundtrip_all_k() {
        for k in 1u32..=22 {
            let m = 1u64 << k;
            let mut l = Label::zero(m);
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
        let mut l = Label::zero(1 << 10);
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
        let mut l = Label::zero(1 << 10);
        l.set(7, 0x3FF);
        l.set(7, 0x123);
        assert_eq!(l.get(7), 0x123);
    }

    // ---- Arithmetic ----

    #[test]
    fn test_add_commutes_with_identity() {
        let mut r = rng();
        for k in [1u32, 4, 10, 22] {
            let m = 1u64 << k;
            let a = rand_label(&mut r, m);
            let b = rand_label(&mut r, m);
            assert_eq!(add(&a, &b), add(&b, &a)); // commutative
            assert_eq!(add(&a, &Label::zero(m)), a); // identity
        }
        // In Z_2, add is XOR, so adding b twice cancels (Free-XOR).
        let a = rand_label(&mut r, 2);
        let b = rand_label(&mut r, 2);
        assert_eq!(add(&add(&a, &b), &b), a);
    }

    #[test]
    fn test_scalar_mul_distributes_over_add() {
        let mut r = rng();
        let m = 1u64 << 10;
        let a = rand_label(&mut r, m);
        let b = rand_label(&mut r, m);
        let s = r.random_range(0..m);
        let lhs = scalar_mul(s, &add(&a, &b));
        let rhs = add(&scalar_mul(s, &a), &scalar_mul(s, &b));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_scalar_mul_z2_zero_and_one() {
        let mut r = rng();
        let a = rand_label(&mut r, 2);
        let z = scalar_mul(0, &a);
        assert_eq!(z, Label::zero(2));
        let o = scalar_mul(1, &a);
        assert_eq!(o, a);
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
}
