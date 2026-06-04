use crate::system::System;
use crate::types::*;

/// Scale a one-hot encoding h by a scalar s.
/// h entries are binary (Z_2), s is in some integer ring R.
/// Output: vector in R where the hot entry equals s, all others 0.
/// Join width: lg|R| bits.
///
/// Output wires inherit `s`'s kind (CF/NCF), so callers can drive the pipeline
/// toward an NCF output by passing an NCF `s` — necessary when the ring happens
/// to be Z_{2^k} but the protocol needs the output labeled as NCF.
pub fn ohe_scale(sys: &mut System, h: &[Wire], s: Wire) -> Vec<Wire> {
    let z = sys.constant_matching(0, s);
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

/// Bulk version of [`ohe_scale`]: scales the same OHE `h` by `S` scalars
/// simultaneously. Returns a `|h| × S` matrix where `out[i][j] = h[i] ? scalars[j] : 0`.
///
/// All scalars must share modulus and kind. When the scalars are NCF and
/// `S > 1`, the per-`h[i]` family of `S` switches (which all share the
/// control wire `h[i]`) is registered as one [`SwitchGroup`]. The garbler/
/// evaluator then derives all members' hashes from a single wide CCRH call,
/// rebating the NCF hash count from `|h|·S` to `|h|·⌈S·lg|R| / λ⌉`.
pub fn ohe_scale_bulk(sys: &mut System, h: &[Wire], scalars: &[Wire]) -> Vec<Vec<Wire>> {
    let s_dim = scalars.len();
    if s_dim == 0 {
        return (0..h.len()).map(|_| Vec::new()).collect();
    }
    let modulus = sys.modulus(scalars[0]);
    let is_cf = sys.is_cf(scalars[0]);
    for &s in scalars {
        assert_eq!(sys.modulus(s), modulus, "ohe_scale_bulk: modulus mismatch");
        assert_eq!(sys.is_cf(s), is_cf, "ohe_scale_bulk: CF/NCF mismatch");
    }

    // Build S parallel ohe_scales, capturing the per-call switch gate ids.
    // cols[j][i] is the i-th output wire for scalar j; switch_gids[j][i] is
    // its switch gate id.
    let mut cols: Vec<Vec<Wire>> = Vec::with_capacity(s_dim);
    let mut switch_gids: Vec<Vec<GateId>> = Vec::with_capacity(s_dim);
    for &s in scalars {
        let z = sys.constant_matching(0, s);
        let mut out = Vec::with_capacity(h.len());
        let mut gids = Vec::with_capacity(h.len());
        let mut acc = z;
        for &h_i in h {
            let pre_gid = sys.num_gates();
            let o = sys.switch(z, h_i);
            out.push(o);
            gids.push(pre_gid);
            acc = sys.add(acc, o);
        }
        sys.join(acc, s);
        cols.push(out);
        switch_gids.push(gids);
    }

    // Register one NCF group per OHE position. Skip when scalars are CF (no
    // packing benefit — each CF switch already costs k hashes regardless) or
    // when S=1 (singleton group is just bookkeeping noise; cost is unchanged).
    if !is_cf && s_dim > 1 {
        for (i, &h_i) in h.iter().enumerate() {
            let members: Vec<GateId> = (0..s_dim).map(|j| switch_gids[j][i]).collect();
            sys.register_ncf_switch_group(h_i, members);
        }
    }

    // Transpose cols (S × |h|) into row-major (|h| × S).
    (0..h.len())
        .map(|i| (0..s_dim).map(|j| cols[j][i]).collect())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Exec;
    use rand::Rng;

    const SAMPLES: usize = 10;
    const MOD: u64 = 16;

    fn rng() -> impl Rng {
        rand::rng()
    }

    #[test]
    fn test_ohe_1bit() {
        for b in 0..2u64 {
            let mut sys = System::new();
            let x = sys.input(2);
            let h = ohe(&mut sys, &[x]);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::new(b, 2));
            exec.run();

            assert_eq!(h.len(), 2);
            for (i, &w) in h.iter().enumerate() {
                let expected = if i as u64 == b { 1 } else { 0 };
                assert_eq!(exec.get(w), Val::new(expected, 2), "ohe_1bit({b})[{i}]");
            }
        }
    }

    #[test]
    fn test_ohe_2bit() {
        for input in 0..4u64 {
            let mut sys = System::new();
            let b0 = sys.input(2);
            let b1 = sys.input(2);
            let h = ohe(&mut sys, &[b0, b1]);

            let mut exec = Exec::new(&sys);
            exec.set(b0, Val::new(input & 1, 2));
            exec.set(b1, Val::new((input >> 1) & 1, 2));
            exec.run();

            assert_eq!(h.len(), 4);
            for (i, &w) in h.iter().enumerate() {
                let expected = if i as u64 == input { 1 } else { 0 };
                assert_eq!(exec.get(w), Val::new(expected, 2), "ohe_2bit({input})[{i}]");
            }
        }
    }

    #[test]
    fn test_ohe_scale_basic() {
        let mut rng = rng();
        for _ in 0..SAMPLES {
            let s_val = rng.random_range(0..MOD);
            let hot_idx = rng.random_range(0..4usize);

            let mut sys = System::new();
            let h: Vec<Wire> = (0..4).map(|_| sys.input(2)).collect();
            let s = sys.input(MOD);
            let out = ohe_scale(&mut sys, &h, s);

            let mut exec = Exec::new(&sys);
            for (i, &w) in h.iter().enumerate() {
                exec.set(w, Val::new(if i == hot_idx { 1 } else { 0 }, 2));
            }
            exec.set(s, Val::new(s_val, MOD));
            exec.run();

            for (i, &w) in out.iter().enumerate() {
                let expected = if i == hot_idx { s_val } else { 0 };
                assert_eq!(exec.get(w), Val::new(expected, MOD));
            }
        }
    }
}
