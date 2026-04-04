//! CRT parameters and reconstruction for evaluating affine maps over a primorial.
//!
//! Given coprime moduli p_1, ..., p_T with primorial M = Π p_i, this module
//! computes the chunking and working-modulus parameters needed to build
//! an affine switch system, and provides CRT reconstruction via Garner's
//! algorithm.

/// Compute gcd of two values.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// CRT parameters derived from an explicit list of primes.
#[derive(Clone, Debug)]
pub struct CrtParams {
    /// Input bit-length (n bits, representing values in [0, 2^n))
    pub n: u32,
    /// The CRT primes p_1, ..., p_T
    pub primes: Vec<u64>,
    /// Number of primes T
    pub num_primes: usize,
    /// Number of input chunks C = ⌈n / chunk_size⌉
    pub num_chunks: usize,
    /// Bits per chunk (= ⌈lg n⌉)
    pub chunk_size: u32,
    /// Working modulus exponent: all intermediate arithmetic lives in Z_{2^ell}
    pub ell: u32,
}

impl CrtParams {
    /// Construct from an explicit list of primes and input bit-length.
    pub fn from_primes(primes: &[u64], n: u32) -> Self {
        assert!(!primes.is_empty());
        assert!(n > 0);

        let primes = primes.to_vec();
        let num_primes = primes.len();

        // Verify pairwise coprimality.
        for i in 0..num_primes {
            assert!(primes[i] > 1, "prime {} is <= 1", primes[i]);
            for j in (i + 1)..num_primes {
                assert_eq!(
                    gcd(primes[i], primes[j]),
                    1,
                    "primes {} and {} are not coprime",
                    primes[i],
                    primes[j]
                );
            }
        }

        // Chunk the n-bit input into ceil(lg n)-bit words.
        let chunk_size = if n <= 1 { 1 } else { (n - 1).ilog2() + 1 };
        let num_chunks = n.div_ceil(chunk_size) as usize;

        // 2^ell must exceed the worst-case residue accumulation:
        //   Σ_c coeff_c * w_c  ≤  C * p_max * (2^chunk_size - 1)
        let p_max = *primes.iter().max().unwrap();
        let max_word = (1u64 << chunk_size) - 1;
        let max_sum = (num_chunks as u128) * (p_max as u128) * (max_word as u128);
        let ell = (max_sum - 1).ilog2() + 2;
        assert!(ell < 64, "working modulus 2^{} exceeds u64", ell);

        CrtParams {
            n,
            primes,
            num_primes,
            num_chunks,
            chunk_size,
            ell,
        }
    }

    /// M = Π p_i
    pub fn primorial(&self) -> u128 {
        self.primes.iter().map(|&p| p as u128).product()
    }
}

/// 2^j mod p via binary exponentiation. Requires p >= 2.
pub fn pow2_mod(j: u32, p: u64) -> u64 {
    assert!(p >= 2, "pow2_mod requires p >= 2, got {}", p);
    let mut result = 1u64;
    let mut base = 2u64 % p;
    let mut exp = j;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % p;
        }
        base = (base * base) % p;
        exp >>= 1;
    }
    result
}

/// Reconstruct x from CRT residues using Garner's mixed-radix algorithm.
/// Overflow can cause panic for M close to 2^128.
pub fn crt_reconstruct(residues: &[u64], primes: &[u64]) -> u128 {
    assert_eq!(residues.len(), primes.len());
    let t = primes.len();
    if t == 0 {
        return 0;
    }

    // Garner's algorithm: find coefficients c_i such that
    //   x = c_0 + c_1·p_0 + c_2·p_0·p_1 + ...
    let mut coeffs: Vec<u128> = vec![0; t];
    coeffs[0] = residues[0] as u128;

    for i in 1..t {
        let p_i = primes[i] as u128;

        // Evaluate partial sum (c_0 + c_1·p_0 + ...) mod p_i
        let mut temp = coeffs[0] % p_i;
        let mut prod = 1u128;
        for j in 1..i {
            prod = mulmod128(prod, primes[j - 1] as u128, p_i);
            temp = (temp + mulmod128(coeffs[j], prod, p_i)) % p_i;
        }

        // c_i = (r_i - temp) · (p_0·...·p_{i-1})^{-1}  mod p_i
        let mut full_prod = 1u128;
        for &p in primes.iter().take(i) {
            full_prod = mulmod128(full_prod, p as u128, p_i);
        }
        let inv = mod_inverse(full_prod, p_i);
        let diff = (p_i + residues[i] as u128 - temp) % p_i;
        coeffs[i] = mulmod128(diff, inv, p_i);
    }

    // Reconstruct from mixed-radix coefficients
    let mut result: u128 = coeffs[0];
    let mut product: u128 = 1;
    for i in 1..t {
        product = product
            .checked_mul(primes[i - 1] as u128)
            .expect("primorial overflow (M > u128)");
        result = result
            .checked_add(coeffs[i].checked_mul(product).expect("CRT overflow"))
            .expect("CRT overflow");
    }
    result
}

/// (a * b) % m, overflow-safe for m < 2^127.
fn mulmod128(a: u128, b: u128, m: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    if a < (1u128 << 64) && b < (1u128 << 64) {
        return (a * b) % m;
    }
    let (mut a, mut b) = (a % m, b % m);
    let mut result = 0u128;
    // Russian peasant multiplication algorithm.
    while b > 0 {
        if b & 1 == 1 {
            result = (result + a) % m;
        }
        a = (a + a) % m;
        b >>= 1;
    }
    result
}

/// Modular inverse via extended Euclidean algorithm.
fn mod_inverse(a: u128, m: u128) -> u128 {
    assert!(m <= i128::MAX as u128, "m too large for i128 arithmetic");
    if m == 1 {
        return 0;
    }
    let a = a % m;
    assert!(a > 0, "no inverse of 0 mod {}", m);
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    assert_eq!(old_r, 1, "gcd({}, {}) = {} ≠ 1", a, m, old_r);
    let m = m as i128;
    (((old_s % m) + m) % m) as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    const SAMPLES: usize = 10;

    fn rng() -> impl Rng {
        rand::rng()
    }

    // ==================== CrtParams construction ====================

    #[test]
    fn test_from_primes_basic() {
        let params = CrtParams::from_primes(&[2, 3, 5, 7, 11, 13, 17, 19, 23, 29], 10);
        assert_eq!(params.primorial(), 6469693230);
        assert_eq!(params.num_primes, 10);
        assert_eq!(params.n, 10);
    }

    #[test]
    fn test_from_primes_chunk_params() {
        let params = CrtParams::from_primes(&[3, 5, 7], 8);
        // chunk_size = ceil(lg 8) = 3
        assert_eq!(params.chunk_size, 3);
        // num_chunks = ceil(8 / 3) = 3
        assert_eq!(params.num_chunks, 3);
        // ell must be large enough and < 64
        assert!(params.ell > params.chunk_size);
        assert!(params.ell < 64);
    }

    #[test]
    fn test_from_primes_n_equals_1() {
        let params = CrtParams::from_primes(&[3, 5], 1);
        assert_eq!(params.chunk_size, 1);
        assert_eq!(params.num_chunks, 1);
    }

    #[test]
    #[should_panic(expected = "not coprime")]
    fn test_from_primes_rejects_non_coprime() {
        CrtParams::from_primes(&[6, 4], 4);
    }

    #[test]
    #[should_panic(expected = "<= 1")]
    fn test_from_primes_rejects_one() {
        CrtParams::from_primes(&[1, 3], 4);
    }

    #[test]
    #[should_panic]
    fn test_from_primes_rejects_empty() {
        CrtParams::from_primes(&[], 4);
    }

    // ==================== primorial ====================

    #[test]
    fn test_primorial() {
        assert_eq!(CrtParams::from_primes(&[2, 3, 5], 4).primorial(), 30);
        assert_eq!(CrtParams::from_primes(&[7, 11], 4).primorial(), 77);
    }

    // ==================== pow2_mod ====================

    #[test]
    fn test_pow2_mod_known() {
        assert_eq!(pow2_mod(0, 5), 1); // 2^0 = 1
        assert_eq!(pow2_mod(3, 5), 3); // 8 mod 5
        assert_eq!(pow2_mod(4, 5), 1); // 16 mod 5
        assert_eq!(pow2_mod(10, 7), 2); // 1024 mod 7
    }

    #[test]
    fn test_pow2_mod_random() {
        let mut rng = rng();
        let primes = [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for _ in 0..SAMPLES {
            let p = primes[rng.random_range(0..primes.len())];
            let j = rng.random_range(0..32u32);
            let expected = (1u64 << j) % p;
            assert_eq!(pow2_mod(j, p), expected, "2^{j} mod {p}");
        }
    }

    // ==================== mulmod128 ====================

    #[test]
    fn test_mulmod128_small() {
        assert_eq!(mulmod128(7, 8, 10), 6);
        assert_eq!(mulmod128(0, 100, 7), 0);
        assert_eq!(mulmod128(100, 0, 7), 0);
    }

    #[test]
    fn test_mulmod128_large() {
        // Values that would overflow a plain u128 multiply
        let a = (1u128 << 100) - 1;
        let b = (1u128 << 100) + 3;
        let m = (1u128 << 101) - 1;
        let result = mulmod128(a, b, m);
        assert!(result < m);
        // a * 1 mod m = a mod m
        assert_eq!(mulmod128(a, 1, m), a % m);
        // commutativity
        assert_eq!(mulmod128(a, b, m), mulmod128(b, a, m));
    }

    #[test]
    fn test_mulmod128_random() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let m = rng.random_range(2..1000u128);
            let a = rng.random_range(0..m);
            let b = rng.random_range(0..m);
            assert_eq!(mulmod128(a, b, m), (a * b) % m, "{a} * {b} mod {m}");
        }
    }

    // ==================== mod_inverse ====================

    #[test]
    fn test_mod_inverse_known() {
        // 3^{-1} mod 7 = 5 (since 3*5 = 15 ≡ 1 mod 7)
        assert_eq!(mod_inverse(3, 7), 5);
        // 2^{-1} mod 5 = 3 (since 2*3 = 6 ≡ 1 mod 5)
        assert_eq!(mod_inverse(2, 5), 3);
    }

    #[test]
    fn test_mod_inverse_random() {
        let mut rng = rng();
        let primes = [3u128, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for _ in 0..SAMPLES {
            let p = primes[rng.random_range(0..primes.len())];
            let a = rng.random_range(1..p);
            let inv = mod_inverse(a, p);
            assert_eq!((a * inv) % p, 1, "{a}^{{-1}} mod {p} = {inv}");
        }
    }

    #[test]
    #[should_panic(expected = "no inverse of 0")]
    fn test_mod_inverse_zero_panics() {
        mod_inverse(0, 7);
    }

    #[test]
    #[should_panic(expected = "≠ 1")]
    fn test_mod_inverse_non_coprime_panics() {
        mod_inverse(4, 6);
    }

    // ==================== crt_reconstruct ====================

    #[test]
    fn test_crt_reconstruct_known() {
        // 17 mod {2,3,5} = {1,2,2}
        assert_eq!(crt_reconstruct(&[1, 2, 2], &[2, 3, 5]), 17);
    }

    #[test]
    fn test_crt_reconstruct_zero() {
        assert_eq!(crt_reconstruct(&[0, 0, 0], &[2, 3, 5]), 0);
    }

    #[test]
    fn test_crt_reconstruct_single_prime() {
        assert_eq!(crt_reconstruct(&[3], &[7]), 3);
    }

    #[test]
    fn test_crt_reconstruct_random_small() {
        let mut rng = rng();
        let primes = [2u64, 3, 5, 7, 11];
        let m: u64 = primes.iter().product(); // 2310
        for _ in 0..SAMPLES {
            let x = rng.random_range(0..m);
            let residues: Vec<u64> = primes.iter().map(|&p| x % p).collect();
            assert_eq!(
                crt_reconstruct(&residues, &primes),
                x as u128,
                "CRT roundtrip for {x}"
            );
        }
    }

    #[test]
    fn test_crt_reconstruct_random_large() {
        let mut rng = rng();
        let primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let m: u128 = primes.iter().map(|&p| p as u128).product(); // 6469693230
        for _ in 0..SAMPLES {
            let x = rng.random_range(0..m as u64);
            let residues: Vec<u64> = primes.iter().map(|&p| x % p).collect();
            assert_eq!(
                crt_reconstruct(&residues, &primes),
                x as u128,
                "CRT roundtrip for {x} with 10 primes"
            );
        }
    }

    #[test]
    fn test_crt_reconstruct_max_residues() {
        // Each residue is p_i - 1; x ≡ -1 mod p_i for all i, so x = M - 1
        let primes = [2u64, 3, 5, 7, 11];
        let residues: Vec<u64> = primes.iter().map(|&p| p - 1).collect();
        let result = crt_reconstruct(&residues, &primes);
        let m: u128 = primes.iter().map(|&p| p as u128).product();
        assert_eq!(result, m - 1);
    }

    // ==================== gcd ====================

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
        assert_eq!(gcd(100, 100), 100);
    }

    #[test]
    fn test_gcd_random() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let a = rng.random_range(1..1000u64);
            let b = rng.random_range(1..1000u64);
            let g = gcd(a, b);
            assert_eq!(a % g, 0, "gcd({a}, {b}) = {g} should divide {a}");
            assert_eq!(b % g, 0, "gcd({a}, {b}) = {g} should divide {b}");
            // Verify it's the *greatest* by checking gcd(a/g, b/g) == 1
            assert_eq!(gcd(a / g, b / g), 1);
        }
    }
}
