# Architecture

`duty-free-bits` garbles and evaluates the switch-system construction `S_aff`:
given an `n`-bit input `x` and per-prime affine coefficients `(a, b)`, it
produces the garbled material a garbler sends an evaluator, and the labels the
evaluator decodes to `a·x + b mod p_i` for each CRT prime `p_i`. Reconstructing
those residues (Chinese Remainder Theorem) recovers `a·x + b` over the
primorial.

The whole computation runs as **straight-line code over bare label words** —
no gate graph, no constraint solver, no worklist. The evaluator knows `x`
(switch-private / data-public), so every derivation order is fixed in advance:
the garbler's pass is x-blind and level-major (it hashes every switch slot),
and the evaluator hashes only the closed slots, solving each hot slot through a
join.

The reference workload is `N = 256`, `S = 6·256 = 1536` affine maps.

---

## 1. The protocol: four straight-line steps

[`build_s_aff`](../src/affine.rs) composes four steps, all straight-line loops
over bare labels:

* **Chunk conversion** ([`comp_gc::extract`](../src/comp_gc/extract.rs)) — one
  per input chunk: pack `lg n` input bits into a ring word `w_c ∈ Z_{2^ℓ}` (a
  one-hot doubling tree over the chunk's bits, width-ℓ casts, a pin join, and
  `Σ p·A_p`).
* **Per prime `p_i`:**
  * **Extract** ([`comp_gc::extract`](../src/comp_gc/extract.rs)) — form
    `r_i = Σ_c coeff_c · w_c` in label space, then run the fused
    `word_to_bin_up` schedule per sub-chunk: the tree, width-`l` casts serving
    as bit-extraction accumulators *and* the peel upcast, and one pin join at
    the class-tree root. Carries out the first sub-chunk's binary one-hot plus
    the remaining bits.
  * **Fold** ([`comp_gc::fold`](../src/comp_gc/fold.rs)) — reduce that one-hot
    mod `p_i` (free Z₂ adds), then fold in the remaining bits to a length-`p_i`
    one-hot `h_p` of `x mod p_i`.
  * **Body** ([`it_gc`](../src/it_gc.rs)) — the information-theoretic GC:
    deliver `a·(x mod p_i) + b` from `h_p`, in `RESIDUE_BATCH_SIZE`-sized
    batches of the `S` maps.

Steps hand `(mask, label)` pairs to each other directly; the Free-XOR offset
`Δ` is global. Peak memory is a few label buffers per step (~8 MB for the
reference workload).

### The two moves the extract/chunk steps are built from

* **The one-hot doubling tree.** Level `m` holds the binary one-hot of the low
  `m` bits. The garbler hashes *every* slot; the evaluator hashes only the
  closed slots and solves the single open one per level through that level's
  1-bit scale join: `y_hot = L_bit ⊕ diff ⊕ (⊕_{j≠hot} y_j)`.
* **Width-`l` casts + the pin join.** Every leaf is cast once to `Z_{2^l}`
  (`A_p = H(leaf_p)`). The single width-`l` pin join `Σ_p A_p ⋈ 1` lets the
  evaluator solve the whole hot class by sums: `L_root = pin_diff` (the
  constant-1 wire's label is 0), so any linear functional `Σ c_p·A_p` with a
  constant coefficient `q` on the unsolved hot class evaluates as
  `Σ_{solved} c_p·A_p + q·(L_root − Σ_{solved} A_p)`.

Residues `x mod 2^i` come from per-level accumulators over the *solved* casts;
they peel the bits LSB-first, and the level-`k` functional `Σ p·A_p` is the
peel upcast — one width-`l` cast layer serves as both the bit extractor and
the upcast (the fusion that `word_to_bin_up` buys over the older
word-to-hot + separate peel).

---

## 2. Storage

* **`Label`** ([`src/label.rs`](../src/label.rs)) — λ = 128 coordinates over
  `Z_{2^k}`. Z₂ labels keep the bit-packed wire format inline (2 u64 words,
  XOR-friendly); k > 1 labels store one coordinate per u32 lane so the λ-wide
  ring ops are plain vectorizable loops. The bit-packed string is the canonical
  wire/hash format; conversion happens only at hash boundaries. Every label is
  control-friendly — the IT-GC body works on bare u64 residues, so there is no
  separate NCF label type.
* **Bare working types** ([`comp_gc::extract`](../src/comp_gc/extract.rs)) —
  inside the hot loops a Z₂ wire is a `[u64; 2]` and a width-`l` wire is a
  `[u32; λ]` lane array; the lane loops run unmasked mod 2^32 (truncation to
  `Z_{2^l}` commutes with wrapping u32 add/mul) with masks applied only at the
  boundaries that make a wire object canonical.

---

## 3. Cryptography

The CCRH core ([`src/crypto`](../src/crypto/mod.rs)) is CCRND over fixed-key
AES-128: `H(x, t) = AES_K(σ(x ⊕ s ⊕ t)) ⊕ σ(x ⊕ s ⊕ t)`. `expand` fills a
buffer in CTR mode; on aarch64 it runs four AES blocks interleaved
(round-major) to keep both AES pipes busy, byte-identical to the one-block path
and to the portable software backend.

[`src/hash.rs`](../src/hash.rs) wraps the core with the label↔block encoding:
`hash_z2` for a bare-word Z₂ switch and `hash_bulk_into` for a switch group
packed into one slab. The extract/chunk steps' width-`l` cast pads use the CTR
`expand` directly and unpack the packed coordinates (NEON gather on aarch64).

**Nonce discipline (paper App. A, Def. 4):** no two CCRH queries may share a
`(domain, id)`. The width-`l` cast hashes own the solo domain (chunk windows
first, then per-prime extract windows, position-indexed); the Z₂ tree hashes
share the bulk domain (bit 63 set) with the fold/body windows, allocating
above `BULK_NONCE_FLOOR = 2^32`. [`affine::NonceLayout`](../src/affine.rs) is
the single source of truth for the window bases and
`test_nonce_windows_disjoint` pins the partition — a collision is otherwise
invisible to output tests, since both parties share the (buggy) layout and
still decode.

---

## 4. Security properties

* **Carry invariant** — every wire satisfies `label = mask + value · Δ_R`.
* **Switches reveal nothing** — the evaluator knows `x` (switch-private /
  data-public), derives every switch control itself, and the garbler sends only
  the join diffs. Communication is exactly the join width.
* **Smudging (paper Thm. 5.2)** — when the evaluator CRT-reconstructs over
  `Z_M`, the garbler must pre-smudge each `b` as `b' = b + μ·q` before deriving
  residues, so the reconstructed integer leaks no more than `a·x + b mod p`.
  This is the caller's responsibility (parameter preparation); the head-to-head
  benchmark includes it in the timed garble side.

What this codebase establishes is correctness and cost, not a security
reduction: the RTCCR-style assumption on the guard hash and join-material
simulatability are proof obligations for the paper, not properties the tests
check.

---

## 5. Performance

On an Apple M1 P-core (single-threaded, `cargo test --release`), `build_s_aff`
runs the reference workload in **~0.030 s/rep**, ~8 MB peak
(`bench_axb_stages`). The full `x + y` application is two such runs.

Per-stage split (ms/rep, garble + eval): body ≈ 16, extract ≈ 9.5, fold ≈ 2,
chunk ≈ 1.5. The steps sit near their AES-block + lane-op floors; the remaining
headroom is protocol-level (fewer MACs/hashes) or cross-prime parallelism, not
execution machinery. Two notes on the hot paths:

* **Body pads are nibble-aligned raw slices** (see `it_gc::pad_bits`):
  extraction is load + shift + mask with no per-pad reduction, trading ~23 %
  more body AES blocks for eliminating the bit-surgery that otherwise dominates
  its non-hash time (−24 % on the body at large S), and shrinking the
  pad-sampling bias from `2^⌈lg p⌉ mod p` to `2^w mod p`.
* **The body is MAC-bound at scale**: it does one multiply-accumulate per
  (slot, member) pair — ~22.5 M per party at `S = 1536`, growing linearly with
  `S` — which a hash-only floor estimate never sees (it undercounts the real
  compute by ~2–3×). At large `S` (e.g. `S = 76800`) the body is ~96 % of the
  run and this MAC count, not hashing, is the dominant cost.

> Measure with hardware counters, not wall time — the M1 throttles under
> ambient load (scaling wall time but not instructions/cycles):
> `N=256 S=1536 /usr/bin/time -l <test-binary> bench_axb_stages --ignored`.

---

## 6. Correctness testing

Run `cargo test` (debug exercises the `debug_assert`s; release exercises the
fast paths). Coverage:

* **Carry invariant per wire** — `test_chunk_label_mask_invariant{,_production}`
  and `test_extract_label_mask_invariant` (in `comp_gc::extract`) pin
  `label = mask + v·Δ` at every emitted wire, exhaustively / randomized at the
  production shapes; `test_fold_label_mask_invariant` does the same for the
  fold.
* **SIMD differentials** — `test_body_batch_simd_matches_scalar_differential`
  (NEON body vs forced-scalar) and `test_unpack_even_k_neon_matches_generic`
  (NEON cast unpack vs the scalar window loop); the CCRH has a golden vector
  plus a portable-vs-NEON equality test.
* **End-to-end known answer** — `test_s_aff_sweep` and
  `test_s_aff_edge_regimes` check the decoded `a·x + b mod p_i` against a direct
  arithmetic oracle across many primes / inputs / `S` and the edge parameter
  regimes.
* **Ledger** — the `Stats` cost fields are checked against the closed forms in
  the step tests (`(2^k − 2) + l·2^k` hashes and `k + l − 1` join bits per
  sub-chunk for extract).

Ignored (manual) benchmarks: `bench_axb_comparison` / `bench_axb_network` (the
field-to-field head-to-head vs the bit-decomposition baseline),
`bench_axb_hashcounts` (per-party CCRH block counts, needs `--features
count-hashes`), `bench_axb_stages` (per-stage split of `build_s_aff`),
`bench_extract_micro` (extract-step primitive attribution), and the CCRH
`bench_ccrnd_single_vs_interleaved`.

---

## 7. Reading order

1. [`label.rs`](../src/label.rs) — the label representation.
2. [`crypto/`](../src/crypto/mod.rs), [`hash.rs`](../src/hash.rs) — the CCRH
   core, nonce rules, and label-aware wrapper.
3. [`comp_gc/extract.rs`](../src/comp_gc/extract.rs) — the one-hot tree + cast
   machinery and the chunk/extract steps (the fused `word_to_bin_up`).
4. [`comp_gc/fold.rs`](../src/comp_gc/fold.rs) — the mod-p fold.
5. [`it_gc.rs`](../src/it_gc.rs) — the IT-GC residue body.
6. [`affine.rs`](../src/affine.rs) — how the steps wire together
   (`build_s_aff`, `Stats`, `NonceLayout`).
7. [`crt/`](../src/crt/mod.rs) — CRT parameters and Garner reconstruction.
8. [`bitdecomp.rs`](../src/bitdecomp.rs) — the switch-free baseline the
   benchmarks compare against.
