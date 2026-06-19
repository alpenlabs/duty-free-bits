# Code tour: from the 2 s implementation to the current one

You last read the codebase at commit `10a1a7e` — the version that took **~2.33 s**
(≈1 s garble + 1 s eval) at N=256, S=1280. This document is a guide to everything
that changed between then and now (**~0.075 s** at N=256, S=1536: ≈25–30 ms garble +
≈45 ms eval). It assumes you understand the protocol and the baseline implementation;
it does *not* re-explain switch systems, the `label = mask + value·Δ_R` carry
invariant, or what a phase is.

> **The one thing to internalize first.** Nothing about the *protocol* changed. Every
> optimization is "compute the identical garbled material with a cheaper representation
> and/or a cheaper schedule." Communication bits, CCRH hash counts, and decoded outputs
> are pinned **bit-for-bit identical** to the original `System`-based implementation by
> tests. So your review is not "is this a correct garbling scheme" (you already vetted
> that) — it's "does each fast path compute the same thing the worklist interpreter
> did." The code is structured to make that question answerable: **every fast path has
> a slow `Label`-path reference still in the tree, and a differential test pinning them
> equal.**

---

## 1. The shape of the change

At 2 s, there was exactly one execution model:

> Each phase builds a `System`. The garbler runs a bidirectional **worklist
> interpreter** over `Vec<Option<Label>>`, recomputing every firable direction of every
> gate on every wakeup, cloning operands, allocating a fresh heap `Vec` per label op.
> The evaluator runs the same interpreter again, plus a cleartext `Exec` worklist pass.
> Labels are bit-packed `Vec<u64>` with per-coordinate bit surgery.

That is clean and obviously correct, but it pays ~50 ns of interpreter + allocator
overhead around every ~1 ns of real arithmetic, three times over.

The current code keeps that interpreter **as the reference**, and adds faster paths
that produce identical results. There are now four execution tiers, fastest first:

| tier | used for | where |
|---|---|---|
| **straight-line kernels** | the two regular sub-circuits (fold, IT-GC body) | `comp_gc/fold.rs`, `it_gc.rs` |
| **compiled VM** | garbling repeated same-shape phases | `comp_gc/arena.rs` (`garble_compiled`) |
| **fused arena eval** | evaluating any arena-eligible phase | `comp_gc/arena.rs` (`fused_eval_arena`) |
| **`Label`-path worklist** | first-of-shape recording, fallback, **and the test oracle** | `comp_gc/garbler.rs`, `comp_gc/evaluator.rs`, `exec.rs` |

A production run of `build_s_aff_streaming` now touches the slow worklist only to
*record* each distinct phase shape once; everything after replays a compiled program
or a flat-arena fused pass.

### The single keystone argument

Every fast path rests on one fact you can check once and then trust everywhere:

> **Label propagation, mask propagation, and cleartext value propagation all flow
> through the *same gate directions*.** A join is resolved via its program diff; a
> switch fires only when its cleartext control is 0; affine gates fire in whatever
> order operands become available. So if you record the order in which the cleartext
> `Exec` pass derived wires, that order is simultaneously a valid schedule for deriving
> masks and a valid schedule for deriving labels — by induction, every operand a step
> reads was written by an earlier step.

That is what licenses (a) recording a garble *schedule* once and replaying it, (b)
fusing value- and label-derivation into one pass, and (c) compiling the schedule to a
typed instruction tape. If you accept this one induction, the rest is mechanical.

---

## 2. Read in this order

The modules build on each other. Suggested path, with the "why" for each:

1. **`label.rs`** — the representation change underneath everything.
2. **`exec.rs`** — the shared `Worklist` and the journal mechanism.
3. **`comp_gc/garbler.rs` & `comp_gc/evaluator.rs`** — the slimmed worklist engines + tape recording.
4. **`comp_gc/fold.rs`** — the simplest kernel; a clean warm-up for the kernel pattern.
5. **`it_gc.rs`** — the body kernel's delayed-reduction + NEON evolution (you saw an early version of this at 2 s).
6. **`comp_gc/arena.rs`** — the big one: flat storage, the compiled garbler, the fused evaluator.
7. **`pipeline.rs`** — how it all gets dispatched (shape keys, the four tiers, nonces).
8. **`affine.rs`** — how the protocol wires the phases together (header is now extract-only; fold + body are kernels).
9. **`crypto/`** — AES instruction-level parallelism + the Def-4 nonce discipline.
10. **`crt/mod.rs`** — `GarnerDecoder` (the decode-side amortization).

Then read the tests last (§5) — they're the proof obligations.

---

## 3. Module-by-module: what changed and why

### 3.1 `label.rs` — dual representation (read first)

**At 2 s:** `CfLabel` was always a bit-packed `Vec<u64>`; `get(i)`/`set(i)` did
shift-and-mask bit surgery per coordinate, and the λ=128-wide ring ops looped calling
those. That cost ~580 ns for a k=22 add — ~220× a raw AES block.

**Now:** `CfLabel` has two in-memory forms (`enum Repr`):
- `Repr::Bits([u64; 2])` for Z₂ (k=1) — inline, no heap; this *is* the wire format.
- `Repr::Lanes(Vec<u32>)` for 2 ≤ k ≤ 32 — one coordinate per u32 lane, so ring ops are
  plain 128-element loops the compiler auto-vectorizes (~18 ns for k=22).

The **wire format is unchanged** (bit-packed, LSB-first); `to_packed_words` /
`from_raw_bits` convert at the boundaries (hashing, serialization). Read
`from_packed_bytes` and `from_raw_bits` together — they're the only places packed↔lane
conversion happens, and there's a round-trip test (`test_packed_words_roundtrip`,
`test_from_packed_bytes_matches_from_raw_bits`) pinning them for every k in 1..=32.

**What to check:** the lane↔packed bit-cursor logic (the `unpack_lanes_words` / pack
loops), and the `const _: () = assert!(LAMBDA == 128 && BITS_WORDS == 2)` that the
inline Z₂ form depends on. `pub(crate) fn lanes` / `from_lanes` are the raw accessors
the arena uses; everything else is as you remember.

### 3.2 `exec.rs` — the shared `Worklist` + journal

**New:** a `Worklist` struct (three bitsets: `known` wires, `done` gates, `queued`
gates) factored out so the garbler, evaluator, and `Exec` all share one demand-driven
engine instead of three copies. The key behavioral change vs 2 s: a gate is **marked
done** once it can never derive a new wire (so it stops being re-popped), wires are
checked via the `known` bitset (cheap) instead of loading the `Option<Label>`, and a
firing gate doesn't re-enqueue itself.

`Exec` additionally grew `run_recorded()` + `journal()`: it records the set-order
`(gate, wire)` list — the "journal" that the keystone argument is about. This is the
discovery pass whose order both replayers reuse.

**What to check:** the `done`-gate rules per gate shape (an over-eager "done" would drop
a still-firable gate). They're mirrored in three files; the comments call this out.

### 3.3 `comp_gc/garbler.rs` & `evaluator.rs` — slimmed engines + recording

**At 2 s:** each gate wakeup recomputed *all three* directions of an `Add`/`Sub` and
cloned all operands up front; `try_set` then discarded the ones already set. Most label
arithmetic was computed and thrown away.

**Now (still the reference path):** each direction is computed only if its target is
unset; operands are borrowed, not cloned; a switch skips its CCRH hash once both sides
are known.

**New entry points** (this is the recording machinery):
- `garble` — the worklist garbler (unchanged in spirit, now demand-driven).
- `garble_recorded` — same, but emits the firing tape (`Vec<JournalEntry>`); used once
  per phase shape. Returns a `Program` *identical* to `garble`'s.
- `garble_replay` — walks a recorded tape linearly, no worklist. **This is the
  `Label`-path reference for the compiled garbler** (and the fallback when the arena
  can't host a shape).
- `replay_with_labels` (evaluator side) — the analogous journal-driven label replay;
  reference for the fused evaluator.

The three garbler entry points share `seed_masks` + `emit_program` helpers so they
can't drift. **Crucial subtlety to check:** `garble_replay`'s validity contract — a
tape is valid only for a System with *identical structure* (same gate kinds, wire ids,
and the same `Mul`-scalar **zero/nonzero pattern**). Scalar *values* may differ; that's
what lets the 79 odd-prime extract phases share one tape. Every replay arm verifies the
taped wire actually belongs to the gate and panics otherwise (a hardening from the
round-3 review — see `test_replay_mismatched_mul_pattern_panics`).

### 3.4 `comp_gc/fold.rs` — the simplest kernel (NEW file)

**At 2 s:** the mod-p OHE fold was built as `System` gates inside the per-prime header
phase — ~60% of every header's gates, all Z₂.

**Now:** `fold_batch_garble` / `fold_batch_eval` execute that fold as straight-line code
on bare `[u64; 2]` label words. Per (bit, slot): one CCRH block + three XORs. The
evaluator recovers the single hidden "hot" pad backward through the join. This file is
short and self-contained — read it as the canonical example of "a regular sub-circuit
lowered to a loop, with the System's semantics reproduced exactly." Its module doc
draws the circuit/algebra correspondence line-by-line.

**What pins it:** `test_fold_kernel_label_mask_invariant` (the carry invariant holds,
exactly one hot slot) and `test_fold_kernel_cost_matches_system_fold` in `tests.rs`
(the fold's communication + hash ledger equals `System::cost` of the equivalent
circuit — i.e. the protocol-visible cost is unchanged).

### 3.5 `it_gc.rs` — the body kernel's evolution

The body kernel (`body_batch_garble`/`body_batch_eval`) existed at 2 s. What changed:
- **Delayed reduction:** pads and weights are < p ≤ 409, so Σpad and Σg·pad accumulate
  raw in u64 (one `% p` per *member* instead of per *slot*) — replaces a hardware
  divide in the inner loop.
- **`extract_pad`:** one unaligned u64 load + shift + compare-subtract, replacing a
  per-bit extraction loop. Byte-identical to `hash::extract_ncf`.
- **`HashSlab`:** one contiguous padded buffer per batch instead of a `Vec<u8>` per
  slot (killed ~270 k allocations/run).
- **`accumulate_pads` → `_scalar` / `_neon`:** the (slot × member) inner loop, ~20 M
  iterations/party, now has a NEON path doing 8 members/iteration (two independent
  4-lane accumulator groups). Scalar remains as the portable path **and** the
  differential oracle.

**What to check:** the `unsafe` NEON block (`accumulate_pads_neon`) — its overflow gate
(`p³ < 2^32` for u32 lanes), the slab's 8-byte trailing pad that keeps the u64 loads
in bounds, and that it falls back to scalar outside the gate. `test_body_batch_simd_
matches_scalar_differential` pins NEON == scalar across 14 primes × 12 batch sizes.

### 3.6 `comp_gc/arena.rs` — the compiled VM + fused eval (NEW file, the big one)

This is where the most scrutiny should go. It has three logical parts:

**(a) Flat storage.** `WireLayout` maps each wire id → a `Slot` (a packed u32: kind +
k + index). `LabelArena` is dense typed storage — `Vec<[u64;2]>` for Z₂ wires,
`Vec<[u32;128]>` for lane wires — persistent across phases. No `Label`, no `Option`, no
per-op allocation during execution. `WireLayout::build` returns `None` for shapes the
arena can't host (NCF wires, k > 32, switch groups), which routes those phases to the
`Label`-path fallback.

**(b) The compiled garbler.** `compile_garble` translates a recorded tape into a typed
instruction stream (`Vec<Instr>` with a small opcode set — read the `OP_*` constants).
**All structural validation happens here, at compile time**: wire-membership per gate
and a definedness simulation over seed-then-tape order. So `garble_compiled` (the hot
path) can skip per-step checks. The one thing it reads from the live `System` is `Mul`
*scalar values* (which legally vary across same-shape phases) — see `OP_MUL_L`/
`OP_MUL_Z2` carrying a gate id, resolved dynamically. This dynamic-scalar point is
load-bearing: baking scalars in was a real bug I caught with the in-pipeline
differential.

**(c) The fused evaluator.** `fused_eval_arena` derives cleartext **values and labels
together** in one worklist pass (`fused_fire` is the per-gate body). This is the
keystone argument cashed out: a direction fires when its value operands are known, and
inductively the label operands are known too. It replaces the old two-stage
"Exec discovery pass → label replay." In CF-only shapes every modulus is 2^k from the
slot, so the value side needs no `Val`/modulus loads.

**What to check:** `compile_garble`'s validation loop (it's the safety net for
`garble_compiled` skipping checks); the `Slot` bit-packing; the NEON `unpack_even_k_neon`
+ its `neon_unpack_ok` gate (k=30 is excluded — its window overflows the 32-bit gather;
the test asserts this). `test_arena_matches_label_path` pins **both** production engines
(`garble_compiled` and `fused_eval_arena`) against the worklist references bit-for-bit
on the real header circuit.

> Note: the file still contains a `garble_tape_arena` / `eval_journal_arena` / `step` /
> `Party` cluster? **No — those were deleted** in the final cleanup (commit `e4e8f38`).
> If you see references to them in old notes, they're gone; production is
> `garble_compiled` + `fused_eval_arena` only.

### 3.7 `pipeline.rs` — dispatch, shape caching, nonces

Read the **module doc first** — it now describes the four execution tiers explicitly.
The key additions vs the 2 s `run_phase`:
- `run_phase_keyed(shape_key, …)`: phases sharing a key build structurally-identical
  Systems. The first records a tape (`garble_recorded`); the second compiles it; the
  rest run `garble_compiled`. A `ShapeCache` per key holds tape + layout + compiled
  program.
- `garble_phase` / `eval_phase`: the two dispatch methods (extracted from one
  200-line block in cleanup). `garble_phase` picks compiled vs replay vs worklist;
  `eval_phase` picks fused-arena vs Exec+replay.
- **`solo_nonce_next`**: each phase reserves `num_gates` fresh solo-domain CCRH nonces
  (the Def-4 fix — see §4).
- **System pooling** (`sys_pool`) and arena reuse across phases (memory churn).
- A `DFB_DIFF=1` env hook: re-derives every compiled phase through `garble_replay` and
  asserts bit-equality. Run the suite with it set to exercise the differential on the
  *real* workload (not just the unit circuits).

There's a hard `assert_eq!((num_gates,num_wires), …)` guarding shape-key collisions
(round-2/3 hardening — see `test_system_extension_after_run_panics`).

### 3.8 `affine.rs` — protocol wiring

**At 2 s:** the per-prime header was one `System` phase doing `r_i` accumulation +
`sub_chunk_extract` + `fold_to_mod_ohe`, then the body kernel.

**Now:** the header is **extract-only** (`r_i` + `sub_chunk_extract`, carrying out the
first sub-chunk OHE + fold bits); the fold runs as a kernel (`fold_batch_*`) and the
body as before. The chunk phases and the odd-prime extract phases use
`run_phase_keyed` (gate-identical shapes → one compiled program each). Read the
per-prime loop to see the three pieces hand off via carries.

**Security-relevant additions to read:** the `KERNEL_NONCE_FLOOR = 2^32` constant and
its window arithmetic (kernels' bulk-domain nonces vs in-System switch-group ids), the
hard assert that windows stay below the bulk-domain flag bit, and the **smudging
contract** documented on `build_s_aff_streaming` (Thm 5.2: `b' = b + μ·p` is the
caller's responsibility before deriving residues).

### 3.9 `crypto/` — AES ILP + nonce freshness

- `crypto/aarch64/mod.rs`: new `ccrnd_wide_with_round_keys::<N>` (round-major N-block
  interleave) + `ccrnd_ctr_fill` drive both M1 AES pipes, taking `expand` from ~4.1 to
  ~1.75 ns/block. `expand` (in `crypto/mod.rs`) now fills whole 4-block groups then a
  single-block tail; the **portable backend is unchanged** and stays byte-identical.
- `hash.rs`: `hash_solo` now takes a `nonce: u64` (was the bare gate id). `hash_z2`
  (new) is the allocation-free single-block CCRH the kernels use. `hash_bulk_into`
  writes into caller storage (the slab) without allocating.
- **Def-4 nonce freshness:** the paper (App. A, Def. 4) requires no two CCRH queries
  share a *nonce* — stricter than the "(seed, nonce) pair" the old comment assumed.
  The pipeline now hands each phase a disjoint nonce window. Solo domain = per-phase
  base + gid (bit 63 clear); bulk domain = bit 63 set, in-System group ids own
  `[0, 2^32)`, kernels above. This is a genuine correctness fix, not just perf.

**What pins it:** `test_ccrh_golden_vectors` (byte-exact CCRH output unchanged),
`test_ccrnd_wide_lanes_match_single`, `test_expand_matches_single_block_oracle`,
`test_portable_backend_matches_neon`.

### 3.10 `crt/mod.rs` — `GarnerDecoder`

`crt_reconstruct` is unchanged and still present. **New:** `GarnerDecoder` precomputes
the Garner prefix products mod p_i and the modular inverses once per prime set, so
decoding S components drops from ~67 ms to ~3 ms. `reconstruct` uses delayed reduction
(raw u64 accumulation, one `% p` per prime); the bound is a **hard** assert in `new`
(decode runs in release). `test_garner_decoder_matches_crt_reconstruct` pins it equal
to the one-shot path.

---

## 4. Security & correctness map (the review checklist)

Protocol-visible quantities are unchanged — here's how each is pinned:

| property | how it's preserved | test(s) |
|---|---|---|
| Garbled material (masks, join diffs) | compiled garbler ≡ worklist garbler | `test_arena_matches_label_path`, `test_garble_recorded_matches_garble`, `DFB_DIFF=1` |
| Output labels / decoded values | fused eval ≡ worklist eval ≡ Exec | `test_arena_matches_label_path`, `test_replay_matches_worklist_eval` |
| Fold kernel correctness | carry invariant + ledger parity | `test_fold_kernel_label_mask_invariant`, `test_fold_kernel_cost_matches_system_fold` |
| Body kernel NEON | NEON ≡ scalar | `test_body_batch_simd_matches_scalar_differential` |
| CCRH output bytes | golden vectors + wide≡single + portable≡NEON | `test_ccrh_golden_vectors`, `test_ccrnd_wide_lanes_match_single`, `test_portable_backend_matches_neon` |
| **Communication (join width)** | ledger parity, printed by scaling test | `test_fold_kernel_cost_matches_system_fold`; `test_s_aff_scaling` output |
| **CCRH nonce freshness (Def-4)** | per-phase disjoint windows; domain split | (argued; window asserts in `affine.rs`/`pipeline.rs`) |
| End-to-end answer | known-answer over many params incl. edge regimes | `test_streaming_sweep`, `test_streaming_edge_regimes`, `test_s_aff_scaling` |

**Where to be skeptical (highest-value review targets):**
1. `comp_gc/arena.rs` `compile_garble` — it's the validation that lets `garble_compiled`
   skip checks. If it under-validates, a bad tape could execute silently.
2. The two `unsafe` NEON blocks (`it_gc.rs accumulate_pads_neon`, `arena.rs
   unpack_even_k_neon`) — bounds + the overflow/width gates.
3. The Def-4 nonce accounting in `pipeline.rs` + `affine.rs` — this is a security
   property argued in prose, not (fully) by test.
4. The `garble_replay` / fused-eval validity contracts — what happens on a
   structurally-mismatched tape/journal (should panic; `test_replay_*_panics`).

---

## 5. The tests are the proof obligations

`cargo test` runs **165 tests** (all green, debug + release); 5 are `#[ignore]`d
(benchmarks + the scaling driver — run them manually). The differential tests in §4 are
the ones that matter for trusting the fast paths; the `#[ignore]`d ones reproduce the
performance numbers:

```sh
# correctness (debug catches the debug_asserts; release exercises the fast paths)
cargo test
cargo test --release
DFB_DIFF=1 cargo test --release        # compiled-garble ≡ replay on every keyed phase

# performance (the corrected workload: N=256, S=6·256=1536; the x+y application is 2×)
N=256 S=1536 cargo test --release test_s_aff_scaling -- --ignored --nocapture
N=256 S=1536 REPS=30 cargo test --release bench_stream_loop -- --ignored --nocapture
cargo test --release bench_primitives -- --ignored --nocapture
```

For per-component instruction/cycle counts (the M1 throttles under load, so trust the
counters, not wall time): run the bare test binary under `/usr/bin/time -l`.

---

## 6. Commit-by-commit changelog (for incremental `git diff`)

The history is linear from `10a1a7e` (your 2 s baseline). Each `perf:`/`fix:` pair is a
round + its adversarial-review hardening. To review incrementally, `git diff A..B` each:

| commit | what landed |
|---|---|
| `55557a3` | **Round 1** (2.33→0.51 s): demand-driven gate firing; body-kernel delayed reduction + word-load `extract_pad`; `CfLabel` u32-lane storage |
| `d5fb5f4` | **Round 2**: `HashSlab` + branchless extract; `Worklist` bitsets; **journal-replay evaluator**; inline `[u64;2]` Z₂ labels; sparse join diffs; subscription CSR; `GarnerDecoder` |
| `9b14b9a` | Round-2 review hardening (stale-CSR-after-extension fix + regression test) |
| `a91fc4c` | **Round 3a**: fold runs as a straight-line kernel (`comp_gc/fold.rs`) |
| `fe4127f` | **Round 3b**: garble-schedule tape record/replay across same-shape phases; NEON body kernel |
| `0c6b574` | Round-3 hardening: replay wire-membership panics; **pre-existing `ell ≡ 1 mod 8` panic fixed**; edge-regime tests |
| `57b146c` | Round-3 docs |
| `50119e3` | Bench harness; workload corrected to S=1536 |
| `f96f9d5` | **Round 4**: flat-arena **compiled-VM garbling** (`comp_gc/arena.rs`); **AES ILP**; **Def-4 nonce freshness**; smudging doc |
| `eaaa606` | Round-4 hardening (arena `add(x,x)` aliasing, nonce window bounds) |
| `0fb7202` | **Round 5**: **fused value+label eval**; NEON even-k unpack; 8-wide MAC kernel |
| `e4e8f38` | **Cleanup**: delete superseded interpreted arena paths; retarget differential test to production paths; doc/rustdoc/clippy clean |

`docs/profiling-s-aff-streaming.md` has the per-round measurements and the
floor analysis (§§9–11 cover rounds 3–5), if you want the "why this was worth it"
narrative alongside the "what changed" you're reading here.

---

## 7. Net effect

- **~31× wall, ~34× instructions, ~26× cycles** vs the 2 s baseline (the workload also
  grew 1280→1536 partway, so the instruction ratio is across a *larger* problem).
- Final composition is ~56% irreducible protocol arithmetic (AES + MACs + λ-lane ring
  ops + pad unpacking); the rest is the one remaining fixpoint (the fused evaluator's
  value-discovery) plus per-phase setup. The remaining gap to a pure hash-only floor is
  named and measured in the profiling doc, not mysterious.
- Line count roughly doubled (≈5.4k → ≈10.9k in `src/`), but most of the growth is the
  two new kernel/VM files (`arena.rs`, `fold.rs`) and tests — the `Label`-path engines
  you already know are intact and now serve as the readable reference for the fast
  paths sitting next to them.
