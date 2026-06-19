//! Portable software AES-128 + CCRND backend.
//!
//! Used on non-aarch64 targets (aarch64 uses the AES-NI backend). Produces
//! byte-identical output to the NEON path — AES is AES — which lets the off-arch
//! build run the *same* CCRH tests. Validated by the FIPS-197 known-answer test
//! plus a direct equality check against the NEON backend (aarch64-only test).
//!
//! Block layout matches the NEON path: a `[u8; 16]` in the standard AES
//! column-major order, with the CCRND orthomorphism σ acting on the two 64-bit
//! halves (bytes `0..8` and `8..16`).
//!
//! NOTE: this table-based AES is **not constant-time**. It exists so the crate
//! compiles and CI runs off-aarch64; the production target is aarch64. A
//! constant-time / bitsliced fallback is future work (see the architecture spec).

/// The 11 AES-128 round keys (round 0 = whitening, rounds 1..=10 the rounds).
pub type RoundKeys = [[u8; 16]; 11];

/// AES S-box.
#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// AES-128 key expansion → 11 round keys (FIPS-197).
pub fn expand_key(key: &[u8; 16]) -> RoundKeys {
    let mut w = [[0u8; 4]; 44];
    for i in 0..4 {
        w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
    }
    for i in 4..44 {
        let mut t = w[i - 1];
        if i % 4 == 0 {
            t = [t[1], t[2], t[3], t[0]]; // RotWord
            for b in t.iter_mut() {
                *b = SBOX[*b as usize]; // SubWord
            }
            t[0] ^= RCON[i / 4 - 1];
        }
        let prev = w[i - 4];
        for (wj, (pj, tj)) in w[i].iter_mut().zip(prev.iter().zip(t.iter())) {
            *wj = pj ^ tj;
        }
    }
    let mut rk = [[0u8; 16]; 11];
    for r in 0..11 {
        for c in 0..4 {
            for b in 0..4 {
                rk[r][4 * c + b] = w[4 * r + c][b];
            }
        }
    }
    rk
}

/// GF(2^8) multiply by x (the AES polynomial).
#[inline]
fn xtime(x: u8) -> u8 {
    (x << 1) ^ (0x1b & (((x >> 7) & 1).wrapping_neg()))
}

#[inline]
fn add_round_key(s: &mut [u8; 16], k: &[u8; 16]) {
    for i in 0..16 {
        s[i] ^= k[i];
    }
}

#[inline]
fn sub_bytes(s: &mut [u8; 16]) {
    for b in s.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

/// ShiftRows on column-major state (byte `r + 4c` is row r, col c).
#[inline]
fn shift_rows(s: &mut [u8; 16]) {
    let o = *s;
    // row r is cyclically left-shifted by r.
    for r in 1..4 {
        for c in 0..4 {
            s[r + 4 * c] = o[r + 4 * ((c + r) % 4)];
        }
    }
}

#[inline]
fn mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let i = 4 * c;
        let a0 = s[i];
        let a1 = s[i + 1];
        let a2 = s[i + 2];
        let a3 = s[i + 3];
        s[i] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
        s[i + 1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
        s[i + 2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
        s[i + 3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
    }
}

/// AES-128 single-block encryption.
fn aes_encrypt(input: [u8; 16], rk: &RoundKeys) -> [u8; 16] {
    let mut s = input;
    add_round_key(&mut s, &rk[0]);
    for round in rk.iter().take(10).skip(1) {
        sub_bytes(&mut s);
        shift_rows(&mut s);
        mix_columns(&mut s);
        add_round_key(&mut s, round);
    }
    sub_bytes(&mut s);
    shift_rows(&mut s);
    add_round_key(&mut s, &rk[10]);
    s
}

#[inline]
fn xor16(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = a[i] ^ b[i];
    }
    o
}

/// Linear orthomorphism σ: `L || R → (L ⊕ R) || L` on the two 64-bit halves.
/// Matches the NEON `vextq`-based σ.
#[inline]
fn sigma(x: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..8 {
        o[i] = x[i] ^ x[i + 8]; // L ⊕ R
        o[i + 8] = x[i]; // L
    }
    o
}

/// CCRND hash: `H(x, t) = AES_K(σ(x ⊕ s ⊕ t)) ⊕ σ(x ⊕ s ⊕ t)`.
pub fn ccrnd(seed: [u8; 16], tweak: [u8; 16], rk: &RoundKeys, public_s: [u8; 16]) -> [u8; 16] {
    let lin = sigma(xor16(xor16(seed, public_s), tweak));
    xor16(aes_encrypt(lin, rk), lin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_fips197_kat() {
        // FIPS-197 Appendix B / C.1 known-answer test.
        let key: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        let pt: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let expected: [u8; 16] = [
            0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
            0xc5, 0x5a,
        ];
        let rk = expand_key(&key);
        assert_eq!(aes_encrypt(pt, &rk), expected);
    }

    #[test]
    fn test_ccrnd_deterministic_and_sensitive() {
        let rk = expand_key(&[0x2b; 16]);
        let s = [0xde; 16];
        let a = ccrnd([0x01; 16], [0x02; 16], &rk, s);
        assert_eq!(a, ccrnd([0x01; 16], [0x02; 16], &rk, s));
        assert_ne!(a, ccrnd([0x09; 16], [0x02; 16], &rk, s)); // distinct seed
        assert_ne!(a, ccrnd([0x01; 16], [0x07; 16], &rk, s)); // distinct tweak
    }
}
