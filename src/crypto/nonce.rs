//! CCRH nonce allocation with Definition-4 legality.
//!
//! Every CCRH call `H(seed, nonce)` must use a fresh `nonce` for a given seed
//! (paper App. A, Def. 4: a query sequence is legal iff no `(seed, nonce)`
//! repeats).

/// Precompute disjoint nonce windows for independent items.
///
/// E.g. the per-prime bodies, which may run in parallel: `bases[i]` is the start
/// of a window of `sizes[i]` nonces, laid out contiguously from `start`. Returned
/// bases never overlap, so each item can draw its nonces without coordination.
pub fn windows(start: u64, sizes: &[u64]) -> Vec<u64> {
    let mut bases = Vec::with_capacity(sizes.len());
    let mut next = start;
    for &size in sizes {
        bases.push(next);
        next = next.checked_add(size).expect("CCRH nonce space exhausted");
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows_are_disjoint_and_cover() {
        let sizes = [4u64, 0, 7, 2];
        let bases = windows(100, &sizes);
        assert_eq!(bases, vec![100, 104, 104, 111]);
        // Adjacent non-empty windows do not overlap.
        for i in 0..sizes.len() {
            for j in (i + 1)..sizes.len() {
                if sizes[i] > 0 && sizes[j] > 0 {
                    let (ai, bi) = (bases[i], bases[i] + sizes[i]);
                    let (aj, bj) = (bases[j], bases[j] + sizes[j]);
                    assert!(bi <= aj || bj <= ai, "windows {i} and {j} overlap");
                }
            }
        }
    }
}
