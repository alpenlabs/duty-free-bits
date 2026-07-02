//! Bit ↔ word ↔ one-hot conversions: the computational-GC
//! sub-circuits that turn the binary input into the per-prime CRT one-hot the
//! IT-GC body consumes.
//!
//! The pipeline composes these: `bin_to_word` packs bits into a ring word,
//! `word_to_bin_up` decomposes a word into bits + binary one-hot + upcast,
//! `sub_chunk_extract` splits a wide word into narrower sub-chunks, and
//! `fold_to_mod_ohe` reduces a one-hot modulo a prime. All build on the
//! primitives in [`super::ohe`].

use super::ohe::{ohe, ohe_scale};
use crate::crt::pow2_mod;
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

/// Fused word → bits + binary one-hot + width-`l` upcast: the straight-line
/// `word_to_hot` (the hand-derived form of the circular Hea24 §4.3 bootstrap).
///
/// Given x ∈ Z_{2^k} and `l ≥ k`, produces:
///
///   * the k bits of x, LSB first;
///   * the 2^k-entry binary one-hot of x (the [`ohe`] doubling-tree leaves);
///   * iff `upcast`, an arithmetic wire holding x in Z_{2^l}.
///
/// One width-`l` cast per one-hot entry serves simultaneously as the
/// bit-extraction accumulator bank and the upcast: a halving tree of
/// congruence-class sums `S_i[q] = Σ_{p ≡ q (2^i)} A[p]` carries the single
/// width-`l` pin join `Σ_p A[p] = 1` at its root, level i's weighted sum
/// encodes `x mod 2^i` (peeling bit i with free ops), and `Σ_p p·A[p]` is the
/// upcast. The hot entry's cast switch is open; the evaluator back-solves it
/// through the pin join and the class tree, exactly the mechanism the
/// worklist engines already resolve.
///
/// Cost: `(2^k − 2) + l·2^k` CF hashes and `k + l − 1` join bits — versus
/// `(2^k − 2) + k·2^k` + a separate width-`l` peel (`l·2^k` hashes,
/// `(2k − 1) + l` join bits total) for the unfused word-to-hot: the k·2^k
/// arithmetic-one-hot layer is fused away.
pub fn word_to_bin_up(
    sys: &mut System,
    x: Wire,
    l: u32,
    upcast: bool,
) -> (Vec<Wire>, Vec<Wire>, Option<Wire>) {
    let m = sys.modulus(x);
    assert!(m.is_power_of_two());
    let k = m.ilog2();
    // Upper bound: the CF label machinery caps ring width at 32 bits
    // (`delta_r` caching, arena lane slots).
    assert!(
        k >= 1 && l >= k && l <= 32,
        "word_to_bin_up requires 1 <= k <= l <= 32"
    );
    let ml = 1u64 << l;

    // Bit wires drive the one-hot tree. Bit 0 is native; bits 1..k are solved
    // through the class tree below (`same_wire` closes the circular schedule).
    let bs: Vec<Wire> = (0..k).map(|_| sys.alloc_wire(2)).collect();
    let b0 = sys.mod2k(x, 1);
    sys.same_wire(bs[0], b0);

    let bin_hot = ohe(sys, &bs);

    // Width-l casts of every one-hot entry (the hot one included — its open
    // switch is back-solved through the pin join).
    let z = sys.constant(0, ml);
    let casts: Vec<Wire> = bin_hot.iter().map(|&bh| sys.switch(z, bh)).collect();

    let up = upcast.then(|| arith_ohe_to_word(sys, &casts));

    // Halving tree of class sums; level i's weighted sum peels bit i.
    let mut acc = casts;
    for i in (1..k).rev() {
        let mid = acc.len() / 2;
        acc = sys.add_vec(&acc[..mid], &acc[mid..]);

        let res = arith_ohe_to_word(sys, &acc); // encodes x mod 2^i, width l
        let res_k = sys.mod2k(res, k);
        let diff = sys.sub(x, res_k); // ≡ 0 (mod 2^i)
        let rem = sys.div2k(diff, i);
        let bit_i = sys.mod2k(rem, 1);
        sys.same_wire(bs[i as usize], bit_i);
    }

    // Root: Σ_p A[p] ⋈ 1 — the single width-l pin join. The halving loop
    // always exits with exactly the two level-1 class sums (k = 1 skips it
    // with the two casts themselves).
    assert_eq!(acc.len(), 2);
    let root = sys.add(acc[0], acc[1]);
    let one = sys.constant(1, ml);
    sys.join(root, one);

    (bs, bin_hot, up)
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
/// Total OHE size is Σ 2^{ℓ_q} instead of 2^ℓ. Each sub-chunk runs the fused
/// [`word_to_bin_up`] with the upcast at the remainder's width, so the peel
/// consumes the gadget's upcast directly — no separate arithmetic one-hot or
/// `ohe_scale` layer.
pub fn sub_chunk_extract(sys: &mut System, r: Wire, sub_widths: &[u32]) -> SubChunkExtraction {
    let m = sys.modulus(r);
    assert!(m.is_power_of_two());
    let ell = m.ilog2();
    assert_eq!(sub_widths.iter().sum::<u32>(), ell);

    let mut remainder = r;
    let mut bits: Vec<Vec<Wire>> = vec![];
    let mut first_bin_hot = vec![];

    for (q, &width) in sub_widths.iter().enumerate() {
        let rem_bits = sys.bitlen(remainder);
        let last = q == sub_widths.len() - 1;

        // Extract sub-chunk: r^(q) = remainder mod 2^width
        let sub = sys.mod2k(remainder, width);

        // Fused decomposition: bits + binary OHE + (unless last) the sub-chunk
        // value upcast to the remainder's width, ready for the peel.
        let l = if last { width } else { rem_bits };
        let (sub_bits, bin_hot, up) = word_to_bin_up(sys, sub, l, !last);

        if q == 0 {
            first_bin_hot = bin_hot;
        }

        bits.push(sub_bits);

        // Peel: remainder = (remainder - sub_word) / 2^width
        if let Some(sub_word) = up {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Exec;
    use rand::{Rng, RngExt};

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

    // ==================== word_to_bin_up ====================

    #[test]
    fn test_word_to_bin_up_exhaustive() {
        // All x at the production points (8,22), (8,14), (6,6) plus edges
        // (k=1, l=k, l>k). Checks bits (LSB first), the full binary OHE
        // (hot entry included), and the upcast value/width.
        for (k, l) in [
            (1u32, 1u32),
            (1, 9),
            (3, 3),
            (4, 8),
            (6, 6),
            (8, 14),
            (8, 22),
        ] {
            let n = 1u64 << k;
            for x_val in 0..n {
                let mut sys = System::new();
                let x = sys.input(n);
                let (bits, bin_hot, up) = word_to_bin_up(&mut sys, x, l, true);

                let mut exec = Exec::new(&sys);
                exec.set(x, Val::new(x_val, n));
                exec.run();

                assert_eq!(bits.len(), k as usize);
                for (i, &b) in bits.iter().enumerate() {
                    assert_eq!(
                        exec.get(b),
                        Val::new((x_val >> i) & 1, 2),
                        "bit {i} for k={k} l={l} x={x_val}"
                    );
                }
                assert_eq!(bin_hot.len(), n as usize);
                for (p, &w) in bin_hot.iter().enumerate() {
                    assert_eq!(
                        exec.get(w),
                        Val::new((p as u64 == x_val) as u64, 2),
                        "bin_hot[{p}] for k={k} l={l} x={x_val}"
                    );
                }
                assert_eq!(
                    exec.get(up.unwrap()),
                    Val::new(x_val, 1u64 << l),
                    "upcast for k={k} l={l} x={x_val}"
                );
            }
        }
    }

    #[test]
    fn test_word_to_bin_up_cost() {
        // The verified ledger: (2^k − 2) + l·2^k CF hashes, k + l − 1 join
        // bits, nothing NCF.
        for (k, l) in [(4u32, 8u32), (6, 6), (8, 14), (8, 22)] {
            let n = 1u64 << k;
            let mut sys = System::new();
            let x = sys.input(n);
            let _ = word_to_bin_up(&mut sys, x, l, true);
            let c = sys.cost();
            assert_eq!(
                c.hash_count_cf as u64,
                (n - 2) + l as u64 * n,
                "hashes for k={k} l={l}"
            );
            assert_eq!(c.join_complexity_cf as u32, k + l - 1, "joins for k={k} l={l}");
            assert_eq!(c.hash_count_ncf, 0);
            assert_eq!(c.join_complexity_ncf, 0);

            // upcast=false (the last-chunk configuration): no wire returned,
            // ledger identical — the upcast is free Mul/Add gates only.
            let mut sys = System::new();
            let x = sys.input(n);
            let (_, _, up) = word_to_bin_up(&mut sys, x, l, false);
            assert!(up.is_none());
            let c2 = sys.cost();
            assert_eq!(c2.hash_count_cf, c.hash_count_cf);
            assert_eq!(c2.join_complexity_cf, c.join_complexity_cf);
        }
    }

    #[test]
    fn test_sub_chunk_extract_cost() {
        // Production shape ell=22, widths [8,8,6]: per-chunk (2^k − 2) + l·2^k
        // = 5886 + 3838 + 446 = 10170 CF hashes and (k + l − 1) = 29 + 21 + 11
        // = 61 join bits per extract (was 14266 / 77 with the unfused path).
        let mut sys = System::new();
        let r = sys.input_bits(22);
        let _ = sub_chunk_extract(&mut sys, r, &[8, 8, 6]);
        let c = sys.cost();
        assert_eq!(c.hash_count_cf, 10170);
        assert_eq!(c.join_complexity_cf, 61);
        assert_eq!(c.hash_count_ncf, 0);
        assert_eq!(c.join_complexity_ncf, 0);
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
}
