# Architecture

`duty-free-bits` garbles and evaluates the switch system `S_aff`: given an
`n`-bit input `x` and per-prime affine coefficients `(a, b)`, it produces the
garbled material a garbler sends an evaluator, and the labels the evaluator
decodes to `a·x + b mod p_i` for each CRT prime `p_i`. Reconstructing those
residues (Chinese Remainder Theorem) recovers `a·x + b` over the primorial.

This document describes how the crate is organized, how garbling and
evaluation execute, the security properties that are preserved, and the
performance characteristics.

The reference workload is `N = 256`, `S = 6·256 = 1536` affine maps.

---

## 1. The production path: four straight-line kernels

`build_s_aff_kernels` ([`src/affine.rs`](../src/affine.rs)) runs the whole
computation as four straight-line kernels over bare label words — **no
`System`, no `Pipeline`, no worklist anywhere on the production path**. The
evaluator knows `x` (switch-private / data-public), so every derivation order
is closed-form: the garbler's pass is x-blind and level-major (hash every
slot), the evaluator hashes only closed switches and solves each hot slot
through a join.

* **Chunk kernel** ([`comp_gc::extract`](../src/comp_gc/extract.rs)) — one
  per input chunk: the straight-line `bin_to_word` (one-hot doubling tree
  over the chunk's bits, width-ℓ casts, pin join, `Σ p·A_p`).
* **Per prime `p_i`:**
  * **Extract kernel** ([`comp_gc::extract`](../src/comp_gc/extract.rs)) —
    `r_i = Σ_c coeff_c · w_c` in label space, then the fused `word_to_bin_up`
    schedule per sub-chunk: tree, width-`l` casts serving as bit-extraction
    accumulators *and* the peel upcast, one pin join at the class-tree root.
  * **Fold kernel** — fold the sub-chunk one-hots into the length-`p_i`
    one-hot `h_p` of `x mod p_i` ([`comp_gc::fold`](../src/comp_gc/fold.rs)).
  * **Body kernel** — the information-theoretic GC: deliver `a·(x mod p_i) + b`
    from `h_p`, in `RESIDUE_BATCH_SIZE`-sized batches of the `S` maps
    ([`it_gc`](../src/it_gc.rs)).

Wires are `[u64; 2]` words (Z₂) and `[u32; λ]` lane arrays (width `l`); the
hot lane loops run unmasked mod 2^32 (truncation to Z_{2^l} commutes) and
masks are applied only at boundaries. Peak memory is a few label buffers per
kernel (~8 MB for the reference workload). The Free-XOR offset `Δ` is global;
masks/labels hand off directly between kernels.

The `System`-phase path (`build_s_aff_streaming` via [`Pipeline`]
(../src/pipeline.rs)) is retained as the reference implementation: the kernels
are differentially tested against it for identical decoded outputs and an
identical cost ledger (§7).

---

## 2. Execution paths (the `System` reference path)

Everything in this section applies to the reference path only — production
runs the kernels of §1. A `System` phase can be garbled and evaluated two
ways. The choice is keyed off a **shape key**: phases sharing a key build
structurally-identical Systems — same gate kinds, wire ids and subscriptions,
and the same `Mul` scalar zero/nonzero pattern (scalar *values* may differ).
The chunk phases share one shape; the odd-prime extract phases share another.

### The keystone

> Cleartext value propagation, mask propagation, and label propagation all
> flow through the **same gate directions**: a join resolves via its program
> diff, a switch fires only when its cleartext control is 0, and affine gates
> fire as their operands become available. So a single firing order is
> simultaneously a valid schedule for deriving masks and for deriving labels —
> by induction, every operand a step reads was produced by an earlier step.

This is what lets the garbler record a firing **schedule** once per shape and
replay it, the compiler lower that schedule to a typed instruction tape, and
the evaluator derive values and labels together in one pass.

### Garbling

* **Worklist** ([`comp_gc::garbler`](../src/comp_gc/garbler.rs)) — a
  demand-driven fixpoint over `Vec<Option<Label>>`. `garble_recorded` runs it
  and records the firing schedule (a `Vec<JournalEntry>`); `garble` runs it
  without recording. The first phase of every shape uses `garble_recorded`;
  unkeyed phases use `garble`.
* **Compiled** ([`comp_gc::arena`](../src/comp_gc/arena.rs)) — `compile_garble`
  lowers a recorded schedule to a typed instruction stream once per shape
  (validating it structurally at compile time); `garble_compiled` then runs
  that program for every phase of the shape, touching the `System` only to read
  `Mul` scalars.

### Evaluation

* **Fused** ([`comp_gc::arena`](../src/comp_gc/arena.rs)) — `fused_eval_arena`
  derives cleartext values and labels together in one worklist pass. Used for
  every keyed, all-CF phase.
* **Journal replay** ([`comp_gc::evaluator`](../src/comp_gc/evaluator.rs)) — a
  cleartext [`Exec`](../src/exec.rs) pass records its derivation journal, and
  `replay_with_labels` walks that journal to derive labels. Used for unkeyed
  (and the rare NCF) phases.

In the reference workload every keyed phase is all-CF, so reference-path
garbling is `garble_recorded` (first of each shape) + `garble_compiled` (the
rest), and its evaluation is `fused_eval_arena` — except the single `p_i = 2`
extract phase, which is unkeyed and uses `garble` + `Exec`/`replay_with_labels`
(keying a one-off phase would only add record + compile overhead).

---

## 3. Storage

* **`System`** ([`src/system.rs`](../src/system.rs)) — wires, gates, constants,
  and the per-wire subscription lists (compiled to a CSR on first use). Tracks
  the circuit's communication + hash cost (`Cost`).
* **`Label`** ([`src/label.rs`](../src/label.rs)) — a `CfLabel` (control-
  friendly, λ = 128 coordinates over Z_{2^k}) or an `NcfLabel` (a single ring
  element). A `CfLabel` keeps Z₂ inline as 2 packed words (also the wire
  format), and k > 1 as one coordinate per u32 lane so the λ-wide ring ops
  vectorize. The bit-packed string is the canonical wire/hash format;
  conversion happens only at hash boundaries.
* **`LabelArena`** ([`comp_gc::arena`](../src/comp_gc/arena.rs)) — dense label
  storage addressed by `Slot` handles (the id-arena pattern): `[u64; 2]` per Z₂
  wire and `[u32; LAMBDA]` per k > 1 wire, in two flat vectors, reused across
  phases. The compiled garbler and fused evaluator run entirely against it — no
  `Box`, no `Option`, no per-op allocation, no enum dispatch in the inner
  loops. A `Label` is materialized only at phase boundaries.

---

## 4. Cryptography

The CCRH core ([`src/crypto`](../src/crypto/mod.rs)) is CCRND over fixed-key
AES-128: `H(x, t) = AES_K(σ(x ⊕ s ⊕ t)) ⊕ σ(x ⊕ s ⊕ t)`. `expand` fills a
buffer in CTR mode; on aarch64 it runs four AES blocks interleaved
(round-major) to keep both AES pipes busy, byte-identical to the one-block path
and to the portable software backend.

[`src/hash.rs`](../src/hash.rs) wraps the core with the label↔block encoding:
`hash_solo` for a single switch, `hash_bulk`/`hash_bulk_into` for a switch
group, and `hash_z2` for the kernels' bare-word Z₂ switches.

**Nonce discipline (paper App. A, Def. 4):** no two CCRH queries may share a
nonce. On the kernel path, the width-`l` cast hashes own the solo domain
(chunk kernels first, then extracts, position-indexed windows) and the Z₂
tree hashes share the bulk domain with the fold/body windows (trees allocate
above them). On the reference path, the `Pipeline` gives each phase a
disjoint window of solo-domain nonces (`solo_nonce_next`, `num_gates` per
phase); the bulk domain (high bit set) is split: in-System switch-group ids
own `[0, 2^32)`, the kernels draw above `KERNEL_NONCE_FLOOR = 2^32`.

---

## 5. Security properties

* **Carry invariant** — every wire satisfies `label = mask + value · Δ_R`
  across phase boundaries.
* **Switches reveal nothing** — the evaluator knows `x` (switch-private /
  data-public), derives every switch control itself, and the garbler sends only
  the join diffs. Communication is exactly the join width.
* **Smudging (paper Thm. 5.2)** — when the evaluator CRT-reconstructs over
  `Z_M`, the garbler must pre-smudge each `b` as `b' = b + μ·p` before deriving
  residues, so the reconstructed integer leaks no more than `a·x + b mod p`.
  This is the caller's responsibility (parameter preparation); see the
  `build_s_aff_streaming` docs.
* **Identical protocol output** — the compiled and fused engines produce
  bit-for-bit the same masks, join diffs, labels, and decoded values as the
  worklist engines, and the same communication-bit and CCRH-hash counts. Tests
  pin this (§7).

---

## 6. Performance

On an Apple M1 P-core (single-threaded, `cargo test --release`), the kernel
path runs the reference workload in **~0.030 s/rep**, ~8 MB peak
(`bench_kernel_loop`). The `System` reference path measures ~0.058 s on the
same binary (`bench_stream_loop`). The full `x + y` application is two such
runs.

Per-stage split of the kernel path (ms/rep, garble + eval): body ≈ 16,
extract ≈ 9.5, fold ≈ 2, chunk ≈ 1.5. The body's pads are nibble-aligned raw
slices (see `it_gc::pad_bits`): extraction is load + shift + mask with no
per-pad reduction, trading ~23 % more body AES blocks for eliminating the
bit-surgery that previously dominated its non-hash time (−24 % on the body
at large S), and shrinking the pad-sampling bias from `2^⌈lg p⌉ mod p` to
`2^w mod p`. The extract/chunk/fold kernels sit near their AES-block +
lane-op floors (a cast measures ~46 ns averaged over the production widths,
against a ~29 ns AES floor at the same mix); the remaining headroom is
protocol-level (fewer MACs/hashes) or cross-prime parallelism, not execution
machinery.

> Measure with hardware counters, not wall time — the M1 throttles under
> ambient load (scaling wall time but not instructions/cycles):
> `N=256 S=1536 /usr/bin/time -l <test-binary> test_s_aff_scaling --ignored`.

### Where the time went on the `System` path (per party)

Shares profiled on the `System` path before the fused `word_to_bin_up`
extract and the kernel port landed — kept for orientation on what the kernel
path eliminated (the fused evaluator's value-discovery fixpoint, the
`System` build + CSR, and per-phase carry materialization):

| component | share | note |
| --- | --- | --- |
| CCRH / AES (4-wide) | ~18 % | at the M1's two-AES-pipe floor (~1.75 ns/block) |
| body-kernel MACs (8-wide NEON) | ~17 % | one multiply-accumulate per (slot, member); ~at floor |
| fused evaluator (worklist + ops) | ~22 % | the one remaining fixpoint, doing value discovery |
| compiled garbler | ~12 % | mostly λ-lane arithmetic |
| hash-output unpack + glue | ~9 % | ~at floor |
| lane arithmetic | ~8 % | real ring work |
| `System` build + CSR | ~8 % | per-phase setup |
| boundaries, fold, misc | ~6 % | |

The kernel path realized the last structurally-addressable item from that
profile: `word_to_bin_up`'s hand-derived straight-line schedule now runs as
the extract/chunk kernels (the `fold.rs` pattern), eliminating the fused
evaluator's value-discovery fixpoint, the per-phase `System` build, and the
carry materialization. What remains is dominated by irreducible protocol
arithmetic — hashing, the IT-GC multiply-accumulates, the λ-lane ring ops,
and pad unpacking; a hash-only floor estimate undercounts the real compute by
~2–3×, because the body alone does one MAC per (slot, member) pair (~20 M per
party) that the hash count never sees.

---

## 7. Correctness testing

Run `cargo test` (debug exercises the `debug_assert`s; release exercises the
fast paths). The fast paths are pinned two ways:

* **Differential** — `test_arena_matches_label_path` (in `comp_gc::arena`)
  asserts the compiled garbler and the fused evaluator agree bit-for-bit with
  the worklist garbler and the journal evaluator on the real header circuit
  (the fused `word_to_bin_up` with its circular class-tree bootstrap, `Mul(0)`
  seeding, switches, joins, Z₂ and lane wires) across primes, inputs, and
  deltas. The NEON kernels have their own
  scalar-vs-SIMD differentials (`test_body_batch_simd_matches_scalar_*`,
  `test_unpack_even_k_neon_matches_generic`), and the CCRH has golden vectors
  plus a portable-vs-NEON equality test.
* **Kernel vs reference** — `test_s_aff_kernels_matches_streaming` runs both
  drivers on identical inputs and asserts identical decoded outputs AND an
  identical cost ledger; `test_chunk_kernel_label_mask_invariant` /
  `test_extract_kernel_label_mask_invariant` (in `comp_gc::extract`) pin the
  kernels' `label = mask + v·Δ` carry invariant per wire, exhaustively /
  randomized at the production shapes.
* **End-to-end known answer** — `test_s_aff_kernels_sweep` (kernel path),
  `test_streaming_sweep`, `test_streaming_edge_regimes`, and (ignored, run
  manually) `test_s_aff_scaling` check the decoded `a·x + b mod p_i` against
  a direct arithmetic oracle, across many primes / inputs / `S` and the edge
  parameter regimes.
* **Cost parity** — `test_fold_kernel_cost_matches_system_fold` pins the fold
  kernel's communication + hash ledger to `System::cost` of the equivalent
  circuit; `test_word_to_bin_up_cost` and `test_sub_chunk_extract_cost` pin
  the extract circuit to its closed-form ledger
  (`(2^k − 2) + l·2^k` hashes, `k + l − 1` join bits per sub-chunk).

Ignored (manual) benchmarks: `bench_kernel_loop` (production path, with the
per-stage split), `bench_stream_loop` (reference path), `bench_extract_micro`
(extract-kernel primitive attribution), `bench_primitives`,
`bench_header_decomposition`, and the CCRH `bench_ccrnd_single_vs_interleaved`.

---

## 8. Reading order

1. [`types.rs`](../src/types.rs), [`system.rs`](../src/system.rs) — wires,
   gates, the constraint graph, cleartext `Val` arithmetic.
2. [`exec.rs`](../src/exec.rs) — the `Worklist` and the cleartext `Exec` pass.
3. [`label.rs`](../src/label.rs) — the dual label representation.
4. [`comp_gc/ohe.rs`](../src/comp_gc/ohe.rs),
   [`comp_gc/convert.rs`](../src/comp_gc/convert.rs) — the Phase-1 circuit
   builders.
5. [`comp_gc/garbler.rs`](../src/comp_gc/garbler.rs),
   [`comp_gc/evaluator.rs`](../src/comp_gc/evaluator.rs) — the worklist garbler
   (and schedule recording) and the journal evaluator.
6. [`comp_gc/extract.rs`](../src/comp_gc/extract.rs),
   [`comp_gc/fold.rs`](../src/comp_gc/fold.rs),
   [`it_gc.rs`](../src/it_gc.rs) — the four straight-line kernels.
7. [`comp_gc/arena.rs`](../src/comp_gc/arena.rs) — flat storage, the compiled
   garbler, the fused evaluator.
8. [`pipeline.rs`](../src/pipeline.rs) — phase orchestration and the
   execution-path dispatch.
9. [`affine.rs`](../src/affine.rs) — how the phases wire together.
10. [`crypto/`](../src/crypto/mod.rs), [`crt/`](../src/crt/mod.rs) — the CCRH
    core and CRT reconstruction.
