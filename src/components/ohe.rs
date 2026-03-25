use crate::types::*;
use crate::system::System;

/// Scale a one-hot encoding h by a scalar s.
/// h entries are binary (Z_2), s is in some integer ring R.
/// Output: vector in R where the hot entry equals s, all others 0.
/// Join width: lg|R| bits.
pub fn ohe_scale(sys: &mut System, h: &[Wire], s: Wire) -> Vec<Wire> {
    let m = sys.modulus(s);
    let z = sys.constant(0, m);
    let mut out = Vec::with_capacity(h.len());

    let mut acc = z;
    for &h_i in h {
        let o = sys.switch(z, h_i);
        out.push(o);
        acc = sys.add(acc, o);
    }
    sys.join(acc, s);

    out
}

/// Build a binary one-hot encoding from binary wires x[0..n].
/// Output: 2^n entries in Z_2.
/// The OHE entry at index bin^{-1}(x) is 1, all others 0.
/// Join width: (n-1) binary joins.
pub fn ohe(sys: &mut System, x: &[Wire]) -> Vec<Wire> {
    assert!(!x.is_empty(), "ohe requires at least one bit");
    let nx0 = sys.not(x[0]);
    let mut h = vec![nx0, x[0]];

    for &xi in &x[1..] {
        let sh = ohe_scale(sys, &h, xi);
        let added = sys.add_vec(&sh, &h);
        h = [added, sh].concat();
    }

    h
}
