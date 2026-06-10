//! Garbled labels.
//!
//! A CF (control-friendly) label on a wire in Z_{2^k} holds LAMBDA
//! coordinates of k bits each. The *wire format* is bit-packed; in memory,
//! Z_2 labels keep that packed form inline while wider labels store one
//! coordinate per u32 lane (see [`CfLabel`]).
//!
//! An NCF label is a single ring element.

/// Security parameter: number of coordinates in a control-friendly label.
pub const LAMBDA: usize = 128;

/// Words in a bit-packed Z_2 label. Inline (no heap): Z_2 labels are the most
/// numerous objects in the system, so `Repr::Bits` stores a fixed array.
const BITS_WORDS: usize = LAMBDA.div_ceil(64);
// The inline representation assumes LAMBDA = 128: exactly 2 words and no
// partial tail word (so packing/unpacking never masks a final word).
const _: () = assert!(LAMBDA == 128 && BITS_WORDS == 2);

/// A control-friendly label: LAMBDA coordinates in Z_{2^k}.
///
/// The *wire format* (communication, hashing) is always the bit-packed string
/// where coordinate `i` occupies bits `[i*k .. (i+1)*k)` LSB-first — see
/// [`CfLabel::from_raw_bits`] / [`CfLabel::to_packed_words`]. In memory, k = 1
/// keeps that packed form (2 words, XOR-friendly), while k > 1 stores one
/// coordinate per u32 lane so the λ-wide ring ops are plain vectorizable loops
/// instead of per-coordinate bit surgery.
#[derive(Clone, Debug)]
pub struct CfLabel {
    repr: Repr,
    /// Ring modulus; must be a power of two in `[2, 2^32]`.
    modulus: u64,
}

/// In-memory coordinate storage (see [`CfLabel`]).
#[derive(Clone, Debug)]
enum Repr {
    /// k == 1: λ bits packed into ⌈λ/64⌉ u64 words (the wire format itself),
    /// stored inline.
    Bits([u64; BITS_WORDS]),
    /// 2 ≤ k ≤ 32: one coordinate per u32 lane, length λ.
    Lanes(Vec<u32>),
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
    /// Zero label in Z_{modulus}. `modulus` must be a power of two in `[2, 2^32]`.
    pub fn zero(modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "CF modulus {} is not power of two",
            modulus
        );
        assert!((2..=(1u64 << 32)).contains(&modulus));
        let k = modulus.trailing_zeros() as usize;
        let repr = if k == 1 {
            Repr::Bits([0u64; BITS_WORDS])
        } else {
            Repr::Lanes(vec![0u32; LAMBDA])
        };
        CfLabel { repr, modulus }
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

    /// The u32 lane array of a k > 1 label (in-memory form; panics on Z_2).
    pub(crate) fn lanes(&self) -> &[u32] {
        match &self.repr {
            Repr::Lanes(l) => l,
            Repr::Bits(_) => panic!("lanes: Z_2 labels are stored bit-packed"),
        }
    }

    /// Build a k > 1 label directly from its lane array.
    pub(crate) fn from_lanes(lanes: Vec<u32>, modulus: u64) -> CfLabel {
        assert!(
            modulus.is_power_of_two() && modulus > 2 && modulus <= (1u64 << 32),
            "from_lanes: modulus {modulus} out of the lanes range"
        );
        assert_eq!(lanes.len(), LAMBDA);
        debug_assert!({
            let mask = coord_mask(modulus.trailing_zeros()) as u32;
            lanes.iter().all(|&l| l & !mask == 0)
        });
        CfLabel {
            repr: Repr::Lanes(lanes),
            modulus,
        }
    }

    /// Raw bit-packed words of a Z_2 label (the wire format). Z_2 only — wider
    /// labels store lanes; serialize those with
    /// [`to_packed_words`](CfLabel::to_packed_words).
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

    /// Build a CF label from the bit-packed wire format. The buffer must have
    /// exactly `⌈LAMBDA · k / 64⌉` words.
    pub fn from_raw_bits(bits: Vec<u64>, modulus: u64) -> Self {
        assert!(
            modulus.is_power_of_two(),
            "CF modulus {} is not power of two",
            modulus
        );
        let k = modulus.trailing_zeros() as usize;
        let words = (LAMBDA * k).div_ceil(64);
        assert_eq!(bits.len(), words, "raw bits length mismatch");
        if k == 1 {
            // LAMBDA = 128 (const-asserted): both words fully used, no tail mask.
            return CfLabel {
                repr: Repr::Bits([bits[0], bits[1]]),
                modulus,
            };
        }
        assert!(k <= 32, "CF modulus 2^{k} exceeds the u32 lane width");
        let mut buf = [0u64; MAX_PACKED_WORDS];
        buf[..words].copy_from_slice(&bits);
        CfLabel {
            repr: Repr::Lanes(unpack_lanes_words(k, &buf)),
            modulus,
        }
    }

    /// Build a CF label directly from packed little-endian bytes (the PRG
    /// output). Equivalent to `from_raw_bits` on the LE-decoded words, with no
    /// intermediate word buffer. The buffer must have exactly
    /// `⌈LAMBDA · k / 64⌉ · 8` bytes.
    pub(crate) fn from_packed_bytes(bytes: &[u8], modulus: u64) -> CfLabel {
        assert!(
            modulus.is_power_of_two(),
            "CF modulus {} is not power of two",
            modulus
        );
        let k = modulus.trailing_zeros() as usize;
        let words = (LAMBDA * k).div_ceil(64);
        assert_eq!(bytes.len(), words * 8, "packed bytes length mismatch");
        let word_at = |w: usize| u64::from_le_bytes(bytes[w * 8..(w + 1) * 8].try_into().unwrap());
        if k == 1 {
            // LAMBDA = 128 (const-asserted): both words fully used, no tail mask.
            return CfLabel {
                repr: Repr::Bits([word_at(0), word_at(1)]),
                modulus,
            };
        }
        assert!(k <= 32, "CF modulus 2^{k} exceeds the u32 lane width");
        // k = 8 (the most frequent hash-output width): lanes are the bytes.
        if k == 8 {
            return CfLabel {
                repr: Repr::Lanes(bytes.iter().map(|&b| b as u32).collect()),
                modulus,
            };
        }
        let mut buf = [0u64; MAX_PACKED_WORDS];
        for (w, slot) in buf.iter_mut().enumerate().take(words) {
            *slot = word_at(w);
        }
        CfLabel {
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

/// Word-wise XOR of two packed-bit Z_2 labels (Z_2 add/sub).
fn xor_words(a: &[u64; BITS_WORDS], b: &[u64; BITS_WORDS]) -> [u64; BITS_WORDS] {
    let mut out = [0u64; BITS_WORDS];
    for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b)) {
        *o = x ^ y;
    }
    out
}

/// Pack bit `bit` of every u32 lane into a Z_2 packed-bit buffer.
/// Accumulates each 64-lane half in a register to avoid per-lane
/// read-modify-write through memory.
fn pack_lane_bit(lanes: &[u32], bit: u32) -> [u64; BITS_WORDS] {
    debug_assert_eq!(lanes.len(), LAMBDA);
    let mut out = [0u64; BITS_WORDS];
    for (w, chunk) in lanes.chunks_exact(64).enumerate() {
        let mut acc = 0u64;
        for (i, &l) in chunk.iter().enumerate() {
            acc |= (((l >> bit) & 1) as u64) << i;
        }
        out[w] = acc;
    }
    out
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

/// Coordinate-wise addition.
pub fn add(x: &Label, y: &Label) -> Label {
    match (x, y) {
        (Label::Cf(a), Label::Cf(b)) => {
            assert_eq!(a.modulus, b.modulus, "add: modulus mismatch");
            let out = match (&a.repr, &b.repr) {
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
            Label::Cf(CfLabel {
                repr: out,
                modulus: a.modulus,
            })
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
            let out = match (&a.repr, &b.repr) {
                // k = 1: subtraction mod 2 is also XOR.
                (Repr::Bits(wa), Repr::Bits(wb)) => Repr::Bits(xor_words(wa, wb)),
                (Repr::Lanes(la), Repr::Lanes(lb)) => {
                    let mask = coord_mask(a.k()) as u32;
                    Repr::Lanes(
                        la.iter()
                            .zip(lb)
                            .map(|(&x, &y)| x.wrapping_sub(y) & mask)
                            .collect(),
                    )
                }
                _ => unreachable!("repr is determined by the (equal) modulus"),
            };
            Label::Cf(CfLabel {
                repr: out,
                modulus: a.modulus,
            })
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

/// Scalar multiplication by `s` (in each coord's ring).
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
            // s ≡ 1 mod 2^k → identity. For k = 1 this covers every nonzero s
            // (the s = 0 path already returned), so only Lanes reach the loop.
            let mask = coord_mask(k);
            if (s & mask) == 1 {
                return Label::Cf(a.clone());
            }
            let Repr::Lanes(la) = &a.repr else {
                unreachable!("k = 1 scalars are 0 or 1 and both returned above");
            };
            let s32 = (s & mask) as u32;
            let lanes = la
                .iter()
                .map(|&x| s32.wrapping_mul(x) & mask as u32)
                .collect();
            Label::Cf(CfLabel {
                repr: Repr::Lanes(lanes),
                modulus: a.modulus,
            })
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
            assert!(k_out >= 1 && k_out <= a.k());
            let out_mod = 1u64 << k_out;
            let repr = match &a.repr {
                Repr::Bits(w) => Repr::Bits(*w), // k_in = k_out = 1
                Repr::Lanes(lanes) if k_out == 1 => Repr::Bits(pack_lane_bit(lanes, 0)),
                Repr::Lanes(lanes) => {
                    let mask = coord_mask(k_out) as u32;
                    Repr::Lanes(lanes.iter().map(|&l| l & mask).collect())
                }
            };
            Label::Cf(CfLabel {
                repr,
                modulus: out_mod,
            })
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
    // k = 0 is the identity (zero low 0 bits, divide by 1); the general CF arm
    // below assumes a Lanes input, which a Z_2 wire (width-1 sub-chunks at
    // ell ≡ 1 mod 8) is not.
    if k == 0 {
        return x.clone();
    }
    match x {
        Label::Cf(a) => {
            assert!(a.modulus > (1u64 << k));
            let out_mod = a.modulus >> k;
            let Repr::Lanes(lanes) = &a.repr else {
                unreachable!("div2k requires modulus > 2, so the input is Lanes");
            };
            let repr = if out_mod == 2 {
                // Wide → Z_2: pack each lane's surviving bit.
                Repr::Bits(pack_lane_bit(lanes, k))
            } else {
                Repr::Lanes(lanes.iter().map(|&l| l >> k).collect())
            };
            Label::Cf(CfLabel {
                repr,
                modulus: out_mod,
            })
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
    fn test_z22_label_is_44_packed_words() {
        // k=22, 128*22 = 2816 bits, ceil / 64 = 44 words on the wire.
        let l = CfLabel::zero(1 << 22);
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
            let l = rand_cf(&mut r, m);
            let packed = l.to_packed_words();
            assert_eq!(packed.len(), (LAMBDA * k as usize).div_ceil(64));
            let back = CfLabel::from_raw_bits(packed, m);
            assert_eq!(back, l, "k={k}");
        }
    }

    #[test]
    fn test_from_packed_bytes_matches_from_raw_bits() {
        // from_packed_bytes(bytes, m) must equal from_raw_bits on the
        // LE-decoded words — it is the same wire format, decoded without the
        // intermediate Vec<u64>. k = 1 pins the Bits arm hash_solo relies on.
        let mut r = rng();
        for k in 1u32..=22 {
            let m = 1u64 << k;
            let words = (LAMBDA * k as usize).div_ceil(64);
            let mut bytes = vec![0u8; words * 8];
            r.fill(&mut bytes[..]);
            let decoded: Vec<u64> = (0..words)
                .map(|w| u64::from_le_bytes(bytes[w * 8..(w + 1) * 8].try_into().unwrap()))
                .collect();
            let via_bytes = CfLabel::from_packed_bytes(&bytes, m);
            let via_words = CfLabel::from_raw_bits(decoded, m);
            assert_eq!(via_bytes, via_words, "k={k}");
        }
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
