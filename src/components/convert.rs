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
    let m = sys.modulus(x);
    assert!(m.is_power_of_two());
    let k = m.ilog2();

    // Allocate k unconstrained binary wires (the bits of x)
    let mut bs = Vec::with_capacity(k as usize);
    for _ in 0..k {
        let b = sys.alloc_wire(2);
        bs.push(b);
    }

    // Build binary OHE from the (as yet unknown) bits
    let bin_hot = ohe(sys, &bs);

    // Promote binary OHE to arithmetic: hot[i] = switch(0_k, bin_hot[i])
    let z = sys.constant(0, m);
    let mut hot = Vec::with_capacity(1usize << k);
    for &bh in &bin_hot {
        hot.push(sys.switch(z, bh));
    }

    // Take successive halves, add them, and use that to constrain bits
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
    hot
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

/// Evaluate an arbitrary function g on a one-hot encoding.
/// Given binary OHE h of x, computes g(x) in Z_{r_mod}.
/// Join width: lg|r_mod| bits.
pub fn hot_to_ring(sys: &mut System, h: &[Wire], truth_table: &[u64], r_mod: u64) -> Wire {
    assert_eq!(h.len(), truth_table.len());

    let one = sys.constant(1, r_mod);
    let sh = ohe_scale(sys, h, one);

    let mut result = sys.constant(0, r_mod);
    for (i, &sh_i) in sh.iter().enumerate() {
        let gi = truth_table[i] % r_mod;
        if gi > 0 {
            let term = sys.mul(gi, sh_i);
            result = sys.add(result, term);
        }
    }
    result
}

/// word_to_ring: given a word x ∈ Z_{2^k}, evaluate g(x) in Z_{r_mod}.
/// Composes word_to_hot with hot_to_ring.
///
/// g: truth table of g: Z_{2^k} → Z_{r_mod}
pub fn word_to_ring(sys: &mut System, x: Wire, truth_table: &[u64], r_mod: u64) -> Wire {
    let m = sys.modulus(x);
    assert!(m.is_power_of_two());
    let k = m.ilog2();
    assert_eq!(truth_table.len(), 1usize << k);

    let hot = word_to_hot(sys, x);

    // Extract binary OHE from the arithmetic one-hot vector.
    // hot[i] is 0 or 1 in Z_{2^k}; mod 2 gives binary indicators.
    let mut bin_hot = Vec::with_capacity(hot.len());
    for &h in &hot {
        let b_i = sys.mod2k(h, 1);
        bin_hot.push(b_i);
    }

    hot_to_ring(sys, &bin_hot, truth_table, r_mod)
}
