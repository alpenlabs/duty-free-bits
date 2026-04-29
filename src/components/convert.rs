use super::crt::pow2_mod;
use super::ohe::{ohe, ohe_scale};
use crate::system::System;
use crate::types::*;

/// Given an arithmetic OHE (entries in Z_M), compute the word encoding:
/// result = Σ (i * h_i)
pub fn arith_ohe_to_word(sys: &mut System, h: &[Wire]) -> Wire {
    assert!(!h.is_empty());
    let m = sys.modulus(h[0]);

    let mut out = sys.constant(0, m);
    for (i, &h_i) in h.iter().enumerate() {
        let scaled = sys.mul(i as u64, h_i);
        out = sys.add(out, scaled);
    }
    out
}

/// Convert a k-bit word x ∈ Z_{2^k} to a 2^k arithmetic one-hot encoding.
///
/// The circular construction from Hea24, Section 4.3.
/// Output: 2^k wires in Z_{2^k} where entry x = 1, all others = 0.
/// Join width: 2k-1 bits.
pub fn word_to_hot(sys: &mut System, x: Wire) -> Vec<Wire> {
    word_to_hot_with_bits(sys, x).0
}

/// Like [`word_to_hot`], but also returns the binary wires (bits of x). We get this for free in the original construction.
pub fn word_to_hot_with_bits(sys: &mut System, x: Wire) -> (Vec<Wire>, Vec<Wire>) {
    let m = sys.modulus(x);
    assert!(m.is_power_of_two());
    let k = m.ilog2();

    let mut bs = Vec::with_capacity(k as usize);
    for _ in 0..k {
        bs.push(sys.alloc_wire(2));
    }

    let bin_hot = ohe(sys, &bs);

    let z = sys.constant(0, m);
    let mut hot = Vec::with_capacity(1usize << k);
    for &bh in &bin_hot {
        hot.push(sys.switch(z, bh));
    }

    let mut acc = hot.clone();
    for i in 1..((k + 1) as usize) {
        let mid = acc.len() / 2;
        acc = sys.add_vec(&acc[..mid], &acc[mid..]);

        let word = arith_ohe_to_word(sys, &acc);
        let diff = sys.sub(x, word);
        let rem = sys.div2k(diff, k - i as u32);
        let mod_rem = sys.mod2k(rem, 1);
        sys.same_wire(bs[k as usize - i], mod_rem);
    }

    let one = sys.constant(1, m);
    sys.join(acc[0], one);
    (hot, bs)
}

/// Convert binary wires (n bits) to an arithmetic word in Z_{2^k}.
/// Join width: (n-1+k) bits.
///
/// k must satisfy k >= n.
pub fn bin_to_word(sys: &mut System, bits: &[Wire], k: u32) -> Wire {
    let n = bits.len();
    assert!(n > 0);
    assert!(
        k as usize >= n,
        "k={} must be >= n={} for bin_to_word",
        k,
        n
    );
    let out_mod = 1u64 << k;

    // Step 1: Build binary OHE of the bit vector
    let bin_hot = ohe(sys, bits);

    // Step 2: Scale by 1 ∈ Z_{2^k} to get arithmetic OHE entries
    let one = sys.constant(1, out_mod);
    let arith_hot = ohe_scale(sys, &bin_hot, one);

    // Step 3: result = Σ_i i * arith_hot[i]
    let mut result = sys.constant(0, out_mod);
    for (i, &h_i) in arith_hot.iter().enumerate() {
        if i > 0 {
            let scaled = sys.mul(i as u64, h_i);
            result = sys.add(result, scaled);
        }
    }

    result
}

/// Evaluate an arbitrary function g on a one-hot encoding, scale by a, add b.
/// Given binary OHE h of x, computes a · g(x) + b in ring R.
///
/// Join width: lg|R| bits.
pub fn hot_to_ring(sys: &mut System, h: &[Wire], truth_table: &[u64], a: Wire, b: Wire) -> Wire {
    let r_mod = sys.modulus(a);
    assert_eq!(sys.modulus(b), r_mod);
    assert_eq!(h.len(), truth_table.len());

    // scale_hot with a: sh_x = a (the hot entry), sh_i = 0 for i ≠ x
    let sh = ohe_scale(sys, h, a);

    // Σ_i g(i) · sh_i = g(x) · a, then add b
    let mut result = sys.constant(0, r_mod);
    for (i, &sh_i) in sh.iter().enumerate() {
        let gi = truth_table[i] % r_mod;
        if gi > 0 {
            let term = sys.mul(gi, sh_i);
            result = sys.add(result, term);
        }
    }
    result = sys.add(result, b);

    result
}

/// word_to_ring: given a word x ∈ Z_{2^k}, evaluate a · g(x) + b in ring R.
/// Composes word_to_hot with hot_to_ring.
pub fn word_to_ring(sys: &mut System, x: Wire, truth_table: &[u64], a: Wire, b: Wire) -> Wire {
    let m = sys.modulus(x);
    assert!(m.is_power_of_two());
    let k = m.ilog2();
    assert_eq!(truth_table.len(), 1usize << k);

    let hot = word_to_hot(sys, x);

    // Extract binary OHE from the arithmetic one-hot vector.
    let mut bin_hot = Vec::with_capacity(hot.len());
    for &h in &hot {
        let b_i = sys.mod2k(h, 1);
        bin_hot.push(b_i);
    }

    hot_to_ring(sys, &bin_hot, truth_table, a, b)
}

// ==================== Sub-chunk extraction ====================

/// Output of Phase 1 (sub-chunk extraction).
#[derive(Debug)]
pub struct SubChunkExtraction {
    /// Bits of each sub-chunk. `bits[q][j]` is the j-th bit (LSB=0) of sub-chunk q.
    pub bits: Vec<Vec<Wire>>,
    /// Binary OHE of the first sub-chunk (2^{ℓ_0} entries in Z_2). Storing these is an optimization.
    pub first_bin_hot: Vec<Wire>,
    /// Sub-chunk widths used.
    pub sub_widths: Vec<u32>,
}

/// Compute sub-chunk widths for decomposing an `ell`-bit value,
/// with each chunk at most `max_width` bits.
pub fn compute_sub_widths(ell: u32, max_width: u32) -> Vec<u32> {
    assert!(ell > 0 && max_width > 0);
    let mut widths = vec![];
    let mut remaining = ell;
    while remaining > 0 {
        let w = remaining.min(max_width);
        widths.push(w);
        remaining -= w;
    }
    widths
}

/// Phase 1: Sub-chunk extraction.
///
/// Decomposes r ∈ Z_{2^ℓ} into sub-chunks of the given widths, extracting
/// the bit decomposition and the first sub-chunk's binary OHE.
///
/// Total OHE size is Σ 2^{ℓ_q} instead of 2^ℓ
pub fn sub_chunk_extract(sys: &mut System, r: Wire, sub_widths: &[u32]) -> SubChunkExtraction {
    let m = sys.modulus(r);
    assert!(m.is_power_of_two());
    let ell = m.ilog2();
    assert_eq!(sub_widths.iter().sum::<u32>(), ell);

    let mut remainder = r;
    let mut bits: Vec<Vec<Wire>> = vec![];
    let mut first_bin_hot = vec![];

    for (q, &width) in sub_widths.iter().enumerate() {
        let rem_mod = sys.modulus(remainder);

        // Extract sub-chunk: r^(q) = remainder mod 2^width
        let sub = sys.mod2k(remainder, width);

        // Decompose sub-chunk via word_to_hot
        let (arith_hot, sub_bits) = word_to_hot_with_bits(sys, sub);

        // Binary OHE from arithmetic OHE (mod 2 of each entry)
        let bin_hot: Vec<Wire> = arith_hot.iter().map(|&h| sys.mod2k(h, 1)).collect();

        if q == 0 {
            first_bin_hot.clone_from(&bin_hot);
        }

        bits.push(sub_bits);

        // Peel: remainder = (remainder - sub_word) / 2^width
        if q < sub_widths.len() - 1 {
            let one_big = sys.constant(1, rem_mod);
            let big_ohe = ohe_scale(sys, &bin_hot, one_big);
            let sub_word = arith_ohe_to_word(sys, &big_ohe);
            let diff = sys.sub(remainder, sub_word);
            remainder = sys.div2k(diff, width);
        }
    }

    SubChunkExtraction {
        bits,
        first_bin_hot,
        sub_widths: sub_widths.to_vec(),
    }
}

/// Phase 2: Fold sub-chunk bits into a binary OHE of (r mod p) of length p.
///
/// Reduces the first sub-chunk's OHE mod p (free additions in Z_2),
/// then folds remaining bits one at a time (p switches + 1 join per bit).
pub fn fold_to_mod_ohe(sys: &mut System, extraction: &SubChunkExtraction, p: u64) -> Vec<Wire> {
    let first_width = extraction.sub_widths[0];

    // Reduce first sub-chunk's binary OHE mod p:
    // h_p[r'] = Σ_{i ≡ r' (mod p)} first_bin_hot[i]
    let mut h_p: Vec<Wire> = (0..p).map(|_| sys.constant(0, 2)).collect();
    for (i, &bh) in extraction.first_bin_hot.iter().enumerate() {
        let r_prime = (i as u64) % p;
        h_p[r_prime as usize] = sys.add(h_p[r_prime as usize], bh);
    }

    // Fold remaining bits from sub-chunks 1, 2, ...
    let mut bit_position = first_width;
    for q in 1..extraction.sub_widths.len() {
        for &bit_wire in &extraction.bits[q] {
            let shift = pow2_mod(bit_position, p);

            // (s, h') = scale1(h): h'[r'] = switch(0, h[r'])
            let zero = sys.constant(0, 2);
            let h_prime: Vec<Wire> = h_p.iter().map(|&hr| sys.switch(zero, hr)).collect();

            // s = Σ h'[r'] (the extracted scalar)
            let mut s = sys.constant(0, 2);
            for &hp in &h_prime {
                s = sys.add(s, hp);
            }

            // s ▷◁ bit_wire (join to constrain)
            sys.join(s, bit_wire);

            // h_new[r'] = h[r'] - h'[r'] + h'[(r' - shift) mod p]
            let mut h_new = Vec::with_capacity(p as usize);
            for r_prime in 0..p as usize {
                let src = ((r_prime as u64 + p - shift) % p) as usize;
                let tmp = sys.add(h_p[r_prime], h_prime[r_prime]);
                h_new.push(sys.add(tmp, h_prime[src]));
            }

            h_p = h_new;
            bit_position += 1;
        }
    }

    h_p
}

/// Evaluate a · (r mod p) + b via sub-chunk extraction and bit folding.
///
/// Equivalent to `word_to_ring` with truth table `[i % p for i in 0..2^ℓ]`,
/// but uses O(Σ 2^{ℓ_q} + (ℓ−ℓ_0)·p) gates instead of O(2^ℓ).
pub fn word_to_ring_mod(
    sys: &mut System,
    r: Wire,
    p: u64,
    sub_widths: &[u32],
    a: Wire,
    b: Wire,
) -> Wire {
    let extraction = sub_chunk_extract(sys, r, sub_widths);
    let h_p = fold_to_mod_ohe(sys, &extraction, p);
    let identity_table: Vec<u64> = (0..p).collect();
    hot_to_ring(sys, &h_p, &identity_table, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Exec;
    use rand::Rng;

    const SAMPLES: usize = 10;

    fn rng() -> impl Rng {
        rand::rng()
    }

    // ==================== arith_ohe_to_word ====================

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

    // ==================== word_to_hot ====================

    #[test]
    fn test_word_to_hot_4bit() {
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
    fn test_word_to_hot_all_values_2bit() {
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

    // ==================== bin_to_word ====================

    #[test]
    fn test_bin_to_word() {
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

    // ==================== hot_to_ring ====================

    #[test]
    fn test_hot_to_ring() {
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
    fn test_hot_to_ring_with_scale_and_offset() {
        let truth_table = vec![0, 1, 2, 3];
        let r_mod = 16u64;

        for input in 0..4u64 {
            let mut sys = System::new();
            let b0 = sys.input(2);
            let b1 = sys.input(2);
            let h = ohe(&mut sys, &[b0, b1]);
            let a = sys.constant(3, r_mod);
            let b = sys.constant(5, r_mod);
            let out = hot_to_ring(&mut sys, &h, &truth_table, a, b);

            let mut exec = Exec::new(&sys);
            exec.set(b0, Val::new(input & 1, 2));
            exec.set(b1, Val::new((input >> 1) & 1, 2));
            exec.run();

            let expected = (3 * input + 5) % r_mod;
            assert_eq!(
                exec.get(out),
                Val::new(expected, r_mod),
                "3·{input} + 5 mod {r_mod}"
            );
        }
    }

    #[test]
    fn test_hot_to_ring_with_random_a_b() {
        let mut rng = rng();
        let truth_table = vec![0, 1, 2, 3];
        let r_mod = 16u64;

        for _ in 0..SAMPLES {
            let a_val = rng.random_range(0..r_mod);
            let b_val = rng.random_range(0..r_mod);
            let input = rng.random_range(0..4u64);

            let mut sys = System::new();
            let b0 = sys.input(2);
            let b1 = sys.input(2);
            let h = ohe(&mut sys, &[b0, b1]);
            let a = sys.constant(a_val, r_mod);
            let b = sys.constant(b_val, r_mod);
            let out = hot_to_ring(&mut sys, &h, &truth_table, a, b);

            let mut exec = Exec::new(&sys);
            exec.set(b0, Val::new(input & 1, 2));
            exec.set(b1, Val::new((input >> 1) & 1, 2));
            exec.run();

            let expected = (a_val * input + b_val) % r_mod;
            assert_eq!(exec.get(out), Val::new(expected, r_mod));
        }
    }

    // ==================== word_to_ring ====================

    #[test]
    fn test_word_to_ring_square() {
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
    fn test_word_to_ring_identity() {
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

    #[test]
    fn test_word_to_ring_with_scale_and_offset() {
        let truth_table: Vec<u64> = (0u64..8).map(|x| (x * x) % 32).collect();
        let r_mod = 32u64;

        for input in 0u64..8 {
            let mut sys = System::new();
            let x = sys.input_bits(3);
            let a = sys.constant(2, r_mod);
            let b = sys.constant(1, r_mod);
            let out = word_to_ring(&mut sys, x, &truth_table, a, b);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, 3));
            exec.run();

            let g = (input * input) % r_mod;
            let expected = (2 * g + 1) % r_mod;
            assert_eq!(
                exec.get(out),
                Val::new(expected, r_mod),
                "2·({input}^2) + 1 mod {r_mod}"
            );
        }
    }

    #[test]
    fn test_word_to_ring_with_random_a_b() {
        let mut rng = rng();
        let truth_table: Vec<u64> = (0u64..4).map(|x| (x * x) % 16).collect();
        let r_mod = 16u64;

        for _ in 0..SAMPLES {
            let a_val = rng.random_range(0..r_mod);
            let b_val = rng.random_range(0..r_mod);
            let input = rng.random_range(0..4u64);

            let mut sys = System::new();
            let x = sys.input_bits(2);
            let a = sys.constant(a_val, r_mod);
            let b = sys.constant(b_val, r_mod);
            let out = word_to_ring(&mut sys, x, &truth_table, a, b);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, 2));
            exec.run();

            let g = (input * input) % r_mod;
            let expected = (a_val * g + b_val) % r_mod;
            assert_eq!(exec.get(out), Val::new(expected, r_mod));
        }
    }

    #[test]
    fn test_word_to_ring_a_zero_returns_b() {
        let truth_table = vec![0, 1, 2, 3];
        let r_mod = 8u64;

        let mut rng = rng();
        for _ in 0..SAMPLES {
            let b_val = rng.random_range(0..r_mod);
            let input = rng.random_range(0..4u64);

            let mut sys = System::new();
            let x = sys.input_bits(2);
            let a = sys.constant(0, r_mod);
            let b = sys.constant(b_val, r_mod);
            let out = word_to_ring(&mut sys, x, &truth_table, a, b);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, 2));
            exec.run();

            assert_eq!(exec.get(out), Val::new(b_val, r_mod));
        }
    }

    // ==================== compute_sub_widths ====================

    #[test]
    fn test_compute_sub_widths_exact_multiple() {
        // 16 bits / max 8 = two chunks of 8
        assert_eq!(compute_sub_widths(16, 8), vec![8, 8]);
    }

    #[test]
    fn test_compute_sub_widths_remainder() {
        // 22 bits / max 8 = [8, 8, 6]
        let widths = compute_sub_widths(22, 8);
        assert_eq!(widths, vec![8, 8, 6]);
        assert_eq!(widths.iter().sum::<u32>(), 22);
    }

    #[test]
    fn test_compute_sub_widths_small() {
        // ell < max_width: single chunk
        assert_eq!(compute_sub_widths(3, 8), vec![3]);
    }

    #[test]
    fn test_compute_sub_widths_one_bit() {
        assert_eq!(compute_sub_widths(1, 8), vec![1]);
    }

    #[test]
    fn test_compute_sub_widths_max_width_one() {
        // Each chunk is 1 bit
        assert_eq!(compute_sub_widths(4, 1), vec![1, 1, 1, 1]);
    }

    // ==================== sub_chunk_extract ====================

    #[test]
    fn test_sub_chunk_extract_bits_reconstruct() {
        // Verify that the extracted bits reconstruct to the original value.
        let ell: u32 = 10;
        let sub_widths = compute_sub_widths(ell, 4); // [4, 4, 2]

        for input in [0u64, 1, 7, 255, (1 << ell) - 1] {
            let mut sys = System::new();
            let x = sys.input_bits(ell);
            let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, ell));
            exec.run();

            // Reconstruct from bits: x = Σ bit[q][j] * 2^(offset + j)
            let mut reconstructed = 0u64;
            let mut bit_pos = 0;
            for (q, width) in sub_widths.iter().enumerate() {
                assert_eq!(extraction.bits[q].len(), *width as usize);
                for j in 0..*width as usize {
                    let bit_val = exec.get(extraction.bits[q][j]).v;
                    assert!(bit_val <= 1, "bit should be 0 or 1");
                    reconstructed |= bit_val << bit_pos;
                    bit_pos += 1;
                }
            }
            assert_eq!(reconstructed, input, "bits should reconstruct to {input}");
        }
    }

    #[test]
    fn test_sub_chunk_extract_first_bin_hot() {
        // Verify the first sub-chunk's binary OHE is correct.
        let ell: u32 = 8;
        let sub_widths = compute_sub_widths(ell, 4); // [4, 4]

        for input in 0u64..(1 << ell) {
            let mut sys = System::new();
            let x = sys.input_bits(ell);
            let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, ell));
            exec.run();

            // First sub-chunk value = input mod 2^4
            let first_sub_val = input & 0xF;
            for (i, &w) in extraction.first_bin_hot.iter().enumerate() {
                let v = exec.get(w).v;
                let expected = if i as u64 == first_sub_val { 1 } else { 0 };
                assert_eq!(v, expected, "first_bin_hot[{i}] for input={input}");
            }
        }
    }

    #[test]
    fn test_sub_chunk_extract_single_chunk() {
        // When ell <= max_width, there's only one sub-chunk (no peeling).
        let ell: u32 = 4;
        let sub_widths = compute_sub_widths(ell, 8); // [4]
        assert_eq!(sub_widths.len(), 1);

        for input in 0u64..(1 << ell) {
            let mut sys = System::new();
            let x = sys.input_bits(ell);
            let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, ell));
            exec.run();

            assert_eq!(extraction.bits.len(), 1);
            assert_eq!(extraction.first_bin_hot.len(), 1 << ell);

            // The binary OHE should be a standard OHE of the full value.
            for (i, &w) in extraction.first_bin_hot.iter().enumerate() {
                let expected = if i as u64 == input { 1 } else { 0 };
                assert_eq!(exec.get(w).v, expected);
            }
        }
    }

    // ==================== fold_to_mod_ohe ====================

    #[test]
    fn test_fold_to_mod_ohe_all_values() {
        // Exhaustively test: for ell=8, p in {3, 5, 7}, verify the OHE
        // has exactly one 1 at position (input mod p).
        let ell: u32 = 8;
        let sub_widths = compute_sub_widths(ell, 4);

        for p in [3u64, 5, 7] {
            for input in 0u64..(1 << ell) {
                let mut sys = System::new();
                let x = sys.input_bits(ell);
                let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);
                let h_p = fold_to_mod_ohe(&mut sys, &extraction, p);

                let mut exec = Exec::new(&sys);
                exec.set(x, Val::from_bits(input, ell));
                exec.run();

                assert_eq!(h_p.len(), p as usize);
                let expected_pos = input % p;
                for (i, &w) in h_p.iter().enumerate() {
                    let v = exec.get(w).v;
                    let expected = if i as u64 == expected_pos { 1 } else { 0 };
                    assert_eq!(
                        v, expected,
                        "p={p}, input={input}: h_p[{i}] = {v}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fold_to_mod_ohe_larger_ell() {
        // Spot-check with ell=12, three sub-chunks [8, 4].
        let ell: u32 = 12;
        let sub_widths = compute_sub_widths(ell, 8);
        let p = 5u64;

        let mut rng = rng();
        for _ in 0..SAMPLES {
            let input = rng.random_range(0u64..(1 << ell));

            let mut sys = System::new();
            let x = sys.input_bits(ell);
            let extraction = sub_chunk_extract(&mut sys, x, &sub_widths);
            let h_p = fold_to_mod_ohe(&mut sys, &extraction, p);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, ell));
            exec.run();

            let expected_pos = input % p;
            for (i, &w) in h_p.iter().enumerate() {
                let v = exec.get(w).v;
                let expected = if i as u64 == expected_pos { 1 } else { 0 };
                assert_eq!(v, expected, "input={input}, h_p[{i}]");
            }
        }
    }

    // ==================== word_to_ring_mod ====================

    #[test]
    fn test_word_to_ring_mod_identity() {
        // word_to_ring_mod with a=1, b=0 should give (input mod p).
        let ell: u32 = 8;
        let sub_widths = compute_sub_widths(ell, 4);

        for p in [3u64, 5, 7] {
            for input in 0u64..(1 << ell) {
                let mut sys = System::new();
                let x = sys.input_bits(ell);
                let a = sys.constant(1, p);
                let b = sys.constant(0, p);
                let out = word_to_ring_mod(&mut sys, x, p, &sub_widths, a, b);

                let mut exec = Exec::new(&sys);
                exec.set(x, Val::from_bits(input, ell));
                exec.run();

                let expected = input % p;
                assert_eq!(exec.get(out), Val::new(expected, p), "p={p}, input={input}");
            }
        }
    }

    #[test]
    fn test_word_to_ring_mod_affine() {
        // Verify a·(input mod p) + b for random a, b.
        let ell: u32 = 10;
        let sub_widths = compute_sub_widths(ell, 4);

        let mut rng = rng();
        for p in [3u64, 5, 7, 11] {
            for _ in 0..SAMPLES {
                let a_val = rng.random_range(0..p);
                let b_val = rng.random_range(0..p);
                let input = rng.random_range(0u64..(1 << ell));

                let mut sys = System::new();
                let x = sys.input_bits(ell);
                let a = sys.constant(a_val, p);
                let b = sys.constant(b_val, p);
                let out = word_to_ring_mod(&mut sys, x, p, &sub_widths, a, b);

                let mut exec = Exec::new(&sys);
                exec.set(x, Val::from_bits(input, ell));
                exec.run();

                let expected = (a_val * (input % p) + b_val) % p;
                assert_eq!(
                    exec.get(out),
                    Val::new(expected, p),
                    "p={p}, a={a_val}, b={b_val}, input={input}"
                );
            }
        }
    }

    #[test]
    fn test_word_to_ring_mod_matches_word_to_ring() {
        // word_to_ring_mod should match word_to_ring with a mod-p truth table.
        let ell: u32 = 6;
        let sub_widths = compute_sub_widths(ell, 4);

        for p in [3u64, 5] {
            let truth_table: Vec<u64> = (0..(1u64 << ell)).map(|i| i % p).collect();

            for input in 0u64..(1 << ell) {
                // word_to_ring_mod path
                let mut sys1 = System::new();
                let x1 = sys1.input_bits(ell);
                let a1 = sys1.constant(1, p);
                let b1 = sys1.constant(0, p);
                let out1 = word_to_ring_mod(&mut sys1, x1, p, &sub_widths, a1, b1);

                let mut exec1 = Exec::new(&sys1);
                exec1.set(x1, Val::from_bits(input, ell));
                exec1.run();

                // word_to_ring path
                let mut sys2 = System::new();
                let x2 = sys2.input_bits(ell);
                let a2 = sys2.constant(1, p);
                let b2 = sys2.constant(0, p);
                let out2 = word_to_ring(&mut sys2, x2, &truth_table, a2, b2);

                let mut exec2 = Exec::new(&sys2);
                exec2.set(x2, Val::from_bits(input, ell));
                exec2.run();

                assert_eq!(
                    exec1.get(out1),
                    exec2.get(out2),
                    "p={p}, input={input}: word_to_ring_mod vs word_to_ring"
                );
            }
        }
    }
}
