# Cost profile: `build_s_aff_streaming` at production scale (N=256, S=1280)

*2026-06-09, Apple M1 (P-core: 128 KB L1d, 12 MB shared L2, ~3.2 GHz), single-threaded,
`cargo test --release`, mimalloc.*

This documents where the 1 s garble / 1 s eval actually went, the experiments
that localized each cost, the optimizations applied (with measured effect), and
the remaining gap to the genuine instruction floor.

**Headline (after two rounds): the stream went 2.33 s → ~0.22 s at matched
clocks — 9.6× fewer CPU cycles, 11.2× fewer instructions — with no protocol or
wire-format changes. Hashing was never the bottleneck; label bookkeeping was.**
§§1–7 document round 1 (2.33 s → 0.51 s); §8 documents round 2
(0.51 s → ~0.22 s) and the updated roadmap.

Interleaved three-way benchmark, identical machine state (clock-independent
counters are the trustworthy columns — this machine throttles under ambient
load, which scales wall time but not instructions/cycles):

| commit | [stream] wall* | instructions | cycles |
|---|---|---|---|
| pre-round-1 (`10a1a7e`) | 3.45 s | 41.40 G | 6.85 G |
| round-1 (`55557a3`) | 0.83 s | 7.36 G | 1.81 G |
| round-2 (HEAD) | 0.36 s | **3.71 G** | **0.71 G** |

*wall at a ~2.0 GHz effective clock; at the M1's full 3.2 GHz the round-2
stream is ~0.20–0.22 s (≈105 ms garble + ≈105 ms eval).

---

## 1. Methodology

1. **Baseline + telemetry.** `test_s_aff_scaling` with `N=256 S=1280`;
   `Pipeline`/`PhaseStats` now record per-phase `build/garble/exec/label-eval`
   wall time, and the IT-GC kernel reports its batches separately
   (`kernel_garble_secs`/`kernel_eval_secs`).
2. **Sampling profiler.** Release build with `CARGO_PROFILE_RELEASE_STRIP=false
   CARGO_PROFILE_RELEASE_DEBUG=2`, sampled with macOS `/usr/bin/sample` at 1 ms.
3. **Microbenchmarks.** `bench_primitives` (ignored test) times every primitive
   the inner loops are made of; `bench_header_decomposition` splits one prime
   header into its `extract` and `fold` sub-circuits as separate pipeline phases.
4. **Hardware counters.** `/usr/bin/time -l` on the bare test binary for
   instructions retired / cycles (child process, not cargo).

Reproduce:

```sh
N=256 S=1280 cargo test --release test_s_aff_scaling -- --ignored --nocapture
cargo test --release bench_primitives -- --ignored --nocapture
cargo test --release bench_header_decomposition -- --ignored --nocapture
```

## 2. Baseline anatomy (the original 2.33 s)

Per-phase wall time (both directions of the protocol):

| phase class            |  n  | garble  | exec   | label-eval | gates  |
|------------------------|-----|---------|--------|------------|--------|
| chunk conversion       | 32  | 0.075 s | 0.003 s| 0.074 s    | 57 k   |
| **prime header**       | 80  | **0.869 s** | 0.075 s| **0.889 s**| **1.39 M** |
| IT-GC body kernel      | 800 | 0.143 s | —      | 0.150 s    | (none) |

Flat profile (top-of-stack, ~1.6 k samples):

| bucket | share | meaning |
|---|---|---|
| `CfLabel::get` | 34 % | per-coordinate bit unpacking in k>1 CF label ops |
| `label::{add,sub,scalar_mul}` | 19 % | the λ=128 coordinate loops (`set` inlined) |
| `body_batch_{garble,eval}` | 15 % | IT-GC kernel: bit-by-bit pad extraction + `%`-division modular ops |
| mimalloc + memmove | 12 % | a fresh heap `Vec` per label op + label clones |
| worklist engine (`garble`, `eval_with_labels`, `Exec`, `try_set`) | 9 % | gate wakeups, queue churn, speculative recomputation |
| u128 `%` (`u128_div_rem`, `__umodti3`) | 3 % | `val_mul` in the cleartext Exec pass; CRT decode (test side) |
| **`hash_solo` / CCRH (AES)** | **1.2 %** | the actual cryptography |

Three structural facts fall out:

* **The per-prime headers are 77 % of runtime and don't scale with S.** They are
  the fixed cost of `r_i` accumulation + `sub_chunk_extract` + `fold_to_mod_ohe`
  per prime (~17.4 k gates each). The part that actually processes all 1280
  affine maps (the IT-GC body) was only 0.29 s. At production S the protocol's
  *variable* cost was already cheap; the *fixed* per-prime cost dominated.
* **Header decomposition:** `extract` is ~16.2 ms per prime *independent of p*
  (×80 ≈ 1.30 s) — all k=8/14/22 CF label arithmetic. `fold` scales with p
  (p=409: 5.8 ms; Σ ≈ 0.19 s) — masses of cheap Z₂ gates plus one 1-block hash
  per switch.
* **Hashing is irrelevant.** ~2.8 M λ-bit CCRH blocks per side, 2.6 ns/block raw
  AES on M1 → ~15 ms/side at the floor, ~1 % of the baseline. The user-visible
  "5 M hash ops × 5 ns = 25 ms" intuition was exactly right — everything else
  was overhead *around* the crypto.

## 3. Primitive costs (microbenchmarks, before → after)

| op | before | after | note |
|---|---|---|---|
| `label::add` CF k=1 | 8.8 ns | 6.4 ns | XOR of 2 words; was already fine |
| `label::add` CF k=22 | 580 ns | **18.6 ns** | was 128 × bit-shift `get`/`set` round trips |
| `label::scalar_mul` CF k=22 | 354 ns | 18.7 ns | |
| `label::mod2k` 22→14 | 321 ns | 16.4 ns | |
| `label::mod2k` 22→1 | 388 ns | 31.5 ns | wide→Z₂ repack, register-accumulated |
| `hash_solo` → CF k=1 | 18 ns | 18 ns | 1 AES block + label wrap |
| raw CCRH block (`expand` 16 B) | 2.6 ns | 2.6 ns | M1 AES pipe |
| `extract_ncf` p=409 | 5.0 ns | ~1–2 ns (kernel-local) | was per-bit loop + hw division |
| `(a*b) % p` u64 | 6.4 ns | mostly eliminated | delayed reduction |
| mimalloc `Vec<u64>` 2/44 words | 5.8 / 8.3 ns | unchanged | allocation is *not* the main tax |

The decisive number: a k=22 label op cost **~220× a raw AES block** before, ~7×
after.

## 4. Optimizations applied (all semantics-preserving, 147/147 tests pass)

Measured on the full workload, cumulative:

| change | total | garble | eval |
|---|---|---|---|
| baseline | 2.33 s | 1.09 | 1.19 |
| **(a) demand-driven gate firing** | 1.36 s | 0.63 | 0.67 |
| **(b) IT-GC kernel: delayed reduction + word-load pad extraction** | 0.84 s | 0.37 | 0.43 |
| (c) `Val` ops: compare-subtract / u64 fast paths (no u128 `%`) | ~same | | (Exec pass is small) |
| **(d) CfLabel: u32-lane storage for k>1** | **0.51 s** | **0.20** | **0.28** |

**(a) Demand-driven gate firing** (`comp_gc/garbler.rs`, `comp_gc/evaluator.rs`).
The bidirectional fixpoint engine recomputed *every* derivable direction of a
gate on *every* wakeup — `Add` computed all three of out/in0/in1 and cloned all
three operand labels up front, with `try_set` then discarding the already-set
ones. Since each gate wakes ~2–4 times, most label arithmetic was computed and
thrown away. Now each direction is computed only if its target wire is still
unset, operands are borrowed instead of cloned, and a switch skips its CCRH
hash entirely once both sides are determined.

**(b) IT-GC body kernel** (`it_gc.rs`). Three changes to the (slot × member)
inner loop, ~17 M iterations per side:
  * *Delayed reduction:* pads and weights are < 2^9, so Σ pad and Σ g·pad
    accumulate raw in u64 (max < 2^27) with a single `% p` per member, replacing
    a `mod_add` + `mod_mul` (hardware divide) per slot. A `debug_assert` pins
    the `p³ < 2^63` bound.
  * *Word-load pad extraction:* `extract_pad` reads the lg p-bit slice with one
    4-byte load + shift instead of a per-bit loop, and reduces with one
    compare-subtract (`acc < 2^⌈lg p⌉ < 2p`) instead of `%`. Byte-identical to
    `hash::extract_ncf`.
  * Loop order flipped to slot-major so each bulk-hash buffer is walked once,
    linearly.

**(d) CfLabel representation** (`label.rs`). The k>1 in-memory representation is
now one coordinate per u32 lane (512 B); k=1 keeps the bit-packed 2-word form
(XOR-fast, and it *is* the wire format the CCRH consumes). Ring ops become
plain 128-lane loops the compiler vectorizes (NEON). The packed bit string
remains the canonical wire/hash format: `from_raw_bits` unpacks once at hash
boundaries, `to_packed_words` serializes (round-trip pinned by a new test).
An intermediate experiment kept packed storage and added sequential
unpack→op→pack cursors — it was **not** faster (~600 ns/op): the per-op
pack/unpack itself is the tax, which is why the representation had to change.
Memory cost: k=22 labels grow 352 B → 512 B; k=8 labels 128 B → 512 B. Peak RSS
moved ~29 MB → ~31 MB. Irrelevant at these scales (see §5).

## 5. Cache and memory behavior (the questions asked)

* **Are we cache-miss bound?** No. `time -l` on the optimized binary:
  7.36 G instructions / 1.77 G cycles = **IPC 4.15** at 3.2 GHz. A memory-bound
  workload on M1 shows IPC ≪ 2; 4+ means the out-of-order core is fed. The cost
  is *instruction count*, not stalls.
* **Working sets.** Largest single phase (p=409 header): ~30.5 k wires →
  masks + labels + values + subscriptions ≈ 7–8 MB. That exceeds L1 (128 KB) by
  ~60× — streaming can't fix that and doesn't need to: it fits the 12 MB L2,
  and L2 latency (~15 cycles) is hidden at this IPC. The kernel's per-batch
  state (p × 144 B bulk-hash buffers + 2 KB accumulators ≈ 60 KB) is
  L1-resident by design — `RESIDUE_BATCH_SIZE=128` is doing its job.
* **Eval's dynamic order.** The LIFO worklist's wire-touch order is effectively
  depth-first along subscription edges — locality is decent, and after (a) each
  gate does O(1) real work per wakeup. No evidence of order-induced thrashing;
  making eval order static (see §6) is worth it for *instruction count*, not
  for cache reasons.
* **Block alignment / array indexing.** The only place alignment mattered was
  the bit-packed label coordinates (lanes straddling u64 words) — eliminated by
  the lane representation — and the lg p-bit pad slices, now read with
  unaligned 4-byte loads (handled in `extract_pad` with a bounded tail copy).
* **The "transpose" for CRT reconstruction** (primes-as-columns →
  per-component residue rows) is *not* a bottleneck: the strided gather is
  ~100 k loads. What *is* measurable in the decode tail is `crt_reconstruct`
  itself: Garner's algorithm is O(t²) `mulmod128` (u128 hardware division)
  per call plus a fresh extended-Euclid `mod_inverse` per (prime, call) —
  ~4 M u128 divmods + 102 k inverses for S=1280, ≈ 60–70 ms, all
  *input-independent* values. (Outside the garble/eval timers; listed in §6.)

## 6. Where the remaining 0.51 s lives, and the road to the floor

Current breakdown (both sides):

| bucket | time | floor estimate | gap mechanism |
|---|---|---|---|
| prime headers (garble+eval) | 0.20 s | ~25 ms | worklist interpreter: ~50 ns/gate of match/queue/wakeup/alloc around ~19 ns of real work |
| IT-GC kernel | 0.20 s | ~80 ms | ~5.8 ns per (slot, member); floor ≈ 2 ns (load+shift+two adds+mul) + 12 ms hashing |
| cleartext Exec pass | 0.065 s | ~15 ms | same interpreter machinery on Copy values |
| chunk phases + build + misc | ~0.05 s | ~10 ms | |
| **total** | **0.51 s** | **~0.13 s** | |

Ranked next steps (none attempted yet):

1. **Replace the fixpoint worklist with a compiled schedule** (biggest remaining
   win, ~0.15–0.2 s). The gate graph is static per phase; the cleartext `Exec`
   pass already determines which direction every gate fires. Record a linear
   instruction tape once (gate id + direction), then garble and eval replay it
   with zero wakeups, zero `Option` checks, and exact-size buffers. This also
   makes eval's order static and prefetch-friendly for free.
2. **In-place / arena label storage** (~50 ms). Every op heap-allocates its
   result (6–8 ns + free + memmove on `try_set`). A bump arena per phase, or
   inline `[u32; 128]` storage in the masks/labels vec, removes ~3 M
   alloc/free pairs and the clone traffic at carry boundaries.
3. **Kernel SIMD pass** (~60 ms). The (slot × member) loop is a 9-bit strided
   gather + multiply-accumulate; NEON `tbl`-based unpacking of 128 pads at a
   time plus vector MLA gets close to the 2 ns/pair floor. Also flatten
   `slot_hash: Vec<Vec<u8>>` into one contiguous buffer (270 k small allocs).
4. **Precompute CRT decode constants** (~60 ms of the tail, trivial). Garner's
   prefix products mod p_i and the `mod_inverse`s depend only on the prime set —
   compute once in `CrtParams`, turning reconstruction into ~t mulmods per
   component with no u128 division and no Euclid.
5. **Share or batch header work across primes** (protocol-level, the real
   prize). `extract` re-derives the bit-decomposition machinery of `r_i` for
   each of 80 primes (~6.8 k gates each, identical shape). If the 80 header
   circuits were garbled as one batch with shared sub-circuits — or the fold's
   p×(ℓ−8) switches bulk-packed into switch groups like the body already does
   (`hash: cf 1.54 M` would collapse) — the fixed cost per prime drops by an
   order of magnitude. Flagged as future work; this is the protocol
   optimization the code comments already anticipate
   (`System::register_ncf_switch_group` is currently unused by the header path:
   the production run reports **0 switch groups**).

## 7. Takeaways

* The protocol's cryptographic core (CCRH/AES) runs at ~15 ms/side at concrete
  parameters; every other cost is representation and interpreter overhead.
  "Get down to the genuine instructions" is therefore an engineering problem,
  not a protocol problem, until ~0.1 s — after which header sharing (§6.5) is
  the only structural lever left.
* The per-prime *fixed* cost (headers) dominated the per-map *variable* cost
  (body) at S=1280 by 6×. Any future scaling argument should quote the two
  separately; `PhaseStats` now does.
* Streaming batching already achieves its memory goal: nothing is DRAM-bound,
  peak phase fits L2, kernel state fits L1, IPC 4.15.

## 8. Round 2: 0.51 s → ~0.22 s (full-clock equivalent)

Round 2 attacked the §6 roadmap. Everything remains semantics-preserving (the
wire format, hash bytes, masks, labels, join diffs and decoded outputs are
unchanged for all inputs); 152 tests pass in debug and release, and the diff
survived an adversarial multi-agent review (4 subsystem reviewers + independent
verifiers re-deriving bounds and running probe tests against both commits).

### What landed

**IT-GC kernel** (`it_gc.rs`, `hash.rs`) — 0.20 s → 0.086 s:
* one contiguous padded `HashSlab` per batch (stride rounded to 8 B, 8 B tail
  pad) instead of a `Vec<u8>` per slot (~270 k allocations gone);
* `extract_pad` is branchless: one unaligned little-endian u64 load + shift +
  lg p-bit mask + compare-subtract. The previous version's variable-length
  tail copy compiled to a branchy memcpy in a ~34 M-iteration loop;
* `hash_bulk_into` writes into caller storage; `hash_bulk` delegates to it, so
  the two cannot drift (golden vectors pinned).

**Worklist engines** (`exec.rs`, `garbler.rs`, `evaluator.rs`) — a shared
`Worklist` with three bitsets: *known-wire* (definedness checks never touch
the 48 B/entry label arrays), *done-gate* (a gate provably unable to derive a
new wire is dropped at pop and never re-enqueued), and *in-queue* (a gate
woken by several wire updates is visited once). Firing gates don't re-enqueue
themselves. The garbler also stopped blanket-seeding all gates — only
subscribers of seeded wires plus `Mul(scalar = 0)` gates (the one shape that
fires with no known input) are seeded.

**Journal replay for the evaluator** — the structural win. The cleartext
`Exec` pass (which must run anyway — it supplies switch controls) records the
set-order journal `(gate, wire)`. Label propagation derives wires through
exactly the same gate directions as value propagation (joins via the program
diff, switches only when ctrl = 0), so the journal **is** a valid label
schedule: `replay_with_labels` re-derives every label on a linear tape with no
queue, no wakeups, and no definedness checks. A differential test pins
replay == worklist on the circular `word_to_hot` + fold circuitry. (The
garbler cannot reuse the journal: masks cross switches unconditionally but do
not cross joins — different reachability.)

**Labels & hashing** (`label.rs`, `hash.rs`):
* Z₂ labels (the most numerous object in the system) store their 2 words
  inline — `Repr::Bits([u64; 2])` — no heap allocation anywhere on a Z₂ path;
* `hash_solo` uses stack buffers and `from_packed_bytes` (PRG bytes → lanes
  directly, branchless u128 windows over a padded word array; k = 8 lanes are
  just the bytes);
* `delta_r` cached per modulus in garble's constant seeding.

**Allocation/structure trims**: sparse sorted join-diff storage in `Program`
(was a zeroed 1.4 MB `Vec<Option<Label>>` per phase for ~2 k joins);
subscriptions as a lazily-compiled CSR (one flat allocation instead of one
`Vec` per wire — build cost halved); `GarnerDecoder` precomputes Garner
prefix products and inverses once per prime set (decode of 1280 components:
67 ms → 3 ms); `Val` ops use compare-subtract / u64 paths (no u128 division).

### Review findings (fixed)

The adversarial review confirmed one real (major) regression — the CSR could
go *silently stale* in release if a `System` was extended after its first
propagation pass, a public-API sequence the old per-wire lists handled
correctly. Fixed: `subscriptions()` hard-asserts the edge count matches the
frozen CSR (loud panic at the next propagation; regression test added).
Minor fixes from the same review: `OnceLock` instead of `OnceCell` (keeps
`System: Sync`), `Exec::run_recorded` clears the journal per run,
`GarnerDecoder::new` hard-asserts its delayed-reduction bound (decode runs in
release; debug-only guards are not enough on public decode paths), replay's
journal-misuse checks promoted to hard asserts, u32-narrowing guards on CSR
build and journal recording, packed-word round-trip coverage extended to
k = 32, and `body_batch_eval` got the same overflow `debug_assert` as garble.

### Where the remaining ~0.22 s (full-clock) lives

| bucket | @2 GHz measured | ≈@3.2 GHz | note |
|---|---|---|---|
| header garble (worklist) | 0.116 s | 72 ms | the last fixpoint engine standing |
| header exec + label replay | 0.114 s | 71 ms | exec discovery + replay math |
| IT-GC kernel | 0.086 s | 54 ms | ~1.6 ns per (slot, member); near scalar floor |
| chunks + build + misc | 0.06 s | 38 ms | |

Updated roadmap, in descending value:
1. **Straight-line header kernels** — garble/eval the extract+fold circuits as
   array code bypassing `System` entirely, the way the body kernel already
   does (~50–70 ms; the remaining interpreter cost is irreducible otherwise).
2. **NEON pad extraction** in the IT-GC kernel — TBL-gather 8 pads per 16 B
   window + vector MLA (~25 ms; unsafe intrinsics + per-lg p tables).
3. **Garbler dry-run + replay** — a bitset-only discovery pass journaling its
   own schedule, then math-only replay (~15–25 ms; superseded by 1).
4. **Thread-level parallelism** — the 80 primes are independent given the
   chunk-word carries; scoped threads over primes would cut wall time ~4–6×
   on M1 without touching single-core cost (orthogonal to all of the above;
   changes the measurement semantics, so kept out of these numbers).

## 9. Round 3: ~0.22 s → 0.11 s, and the distance to the theoretical floor

Round 3 removed the `System` interpreter from everything regular and
vectorized the body kernel. Interleaved four-way benchmark (identical machine
state, full clock — the run reproduced the session's original wall numbers):

| commit | [stream] wall | instructions | cycles |
|---|---|---|---|
| pre-round-1 (`10a1a7e`) | 2.10–2.33 s | 41.39 G | 6.69–6.89 G |
| round-1 (`55557a3`) | 0.49–0.57 s | 7.36 G | 1.78 G |
| round-2 (`9b14b9a`) | 0.22–0.25 s | 3.71 G | 0.71 G |
| round-3 (HEAD) | **0.11–0.12 s** | **1.78 G** | **0.36 G** |

**Cumulative: 19× wall, 18.7× cycles, 23× instructions.** The original
"1 s garble + 1 s eval" is now ≈50 ms garble + ≈60 ms eval.

### What landed

* **Fold kernel** (`comp_gc/fold.rs`): the mod-p OHE fold — the regular,
  Z₂-only ~60 % of every header circuit — garbles/evaluates as straight-line
  code on bare `[u64; 2]` label words, like the body kernel. Per (bit, slot):
  one CCRH block + three XORs; the evaluator recovers the single hot pad
  backward through the join (`L_h'[hot] = L_bit ⊕ diff ⊕ ⊕_{r≠hot} pads`).
  Fold ids draw from per-prime bulk-domain nonce windows above
  `KERNEL_NONCE_FLOOR = 2^32` (the `[0, 2^32)` range stays reserved for
  in-System switch-group ids). Cost-ledger parity with the System path is
  asserted by a dedicated test (CF joins and CF hashes identical; the
  `total_program_bits` telemetry changed meaning — extract output masks
  replace h_p masks — and is not a parity quantity). Fold kernel wall time:
  ~1 ms per side for all 80 primes.
* **Garble-schedule replay** (`garbler.rs` + `pipeline.run_phase_keyed`): the
  garbler's firing schedule depends only on circuit structure, so the 79
  odd-prime extract phases (gate-identical; only Mul scalar values differ)
  and the 32 chunk phases replay one recorded tape with no worklist. The
  validity contract (identical structure incl. the Mul zero/nonzero pattern)
  is documented and enforced by per-arm wire-membership panics; p = 2 (whose
  zero coefficients skip gates) garbles unkeyed.
* **NEON body kernel** (`it_gc.rs`): 4 members per `uint32x4` group,
  member-major with register accumulators — one unaligned u64 load per
  (slot, group), per-lane `vshlq` alignment, vectorized compare-subtract,
  `vmlaq_n_u32`. Gated on `p³ < 2^32` (u32-lane overflow bound, admits
  p ≤ 1625) with the scalar path as portable fallback and differential
  oracle. Body kernel: ~0.6 ns per (slot, member) pair.

### Review findings (fixed)

A second adversarial panel (4 reviewers + verifiers, with reproductions)
confirmed two majors, both fixed: (1) `garble_replay`/`replay_with_labels`
arms did not verify the taped wire belongs to the gate — a foreign tape could
silently fabricate a mask (reproduced); every arm now fails loudly, and the
pipeline's shape check is a hard assert. (2) A **pre-existing** crash (present
since before round 1): any parameter set with ell ≡ 1 (mod 8) — e.g. the
production prime set at n = 32 — panicked in `label::div2k` on the width-1
sub-chunk; `div2k(_, 0)` is now the identity, with an end-to-end regression
test. Hardenings: hard asserts on `x_bits` validity, the `r_i < 2^ell` bound,
and the kernel delayed-reduction bound; edge-regime sweep tests
(p > 2^first_width, fold_bits = 0, short last chunk, single-prime sets).

### Distance to the theoretical floor

Counting *significant* instructions for this protocol at these parameters
(both parties, full run):

| component | count | ≈instructions |
|---|---|---|
| CCRH/AES (5.55 M λ-blocks, 2 parties) | 5.55 M × ~22 | ~120 M |
| body kernel MACs (34.4 M pairs, NEON ÷4) | 8.6 M × ~10 | ~90 M |
| extract lane ops (1.1 M × λ=128 u32 lanes) | 1.1 M × ~40 | ~45 M |
| fold/Z₂ XORs, exec values, tapes, build, decode | | ~120 M |
| **floor** | | **~0.4 G** |

Measured: 1.78 G — **~4× above the irreducible floor** (0.36 G cycles ≈
110 ms ≈ 3× above the ~40 ms cycle floor, since AES and NEON sustain
higher IPC than the rest). The remaining slack is concentrated in the extract
phases' per-op abstraction (enum dispatch + `Option` wrapping + one heap
`Vec<u32>` per lane-op result ≈ 3× the bare lane math), the `Exec` value
worklist, and `System` construction. Closing it requires the full
compiled-VM design: flat per-wire label arenas (no enum, no `Option`, no
per-op allocation) executed by the recorded tapes — a representation rewrite
of `label.rs`/`garbler.rs`/`evaluator.rs` with modest absolute payoff
(~70 ms → ~45 ms) and high churn. Orthogonally, the 80 primes are
independent given the chunk-word carries: scoped threads would cut wall time
another ~4–6× on this machine without changing per-core cost.

### Updated takeaways

* The protocol now runs within ~3–4× of its instruction-counted floor; the
  cryptography (AES) is ~7 % of remaining cycles, the two hand-written
  kernels ~30 %, and the residual `System` machinery the rest.
* Every regular sub-circuit (fold, body) pays ~zero interpretation cost; the
  only circuits still interpreted are the genuinely irregular ones
  (`word_to_hot`'s circular construction), and those replay recorded
  schedules across structurally-identical phases.
* All protocol-visible quantities — communication bits, CF/NCF hash counts,
  decoded outputs — are pinned identical to the original System-only
  implementation by ledger-parity and known-answer tests.

## 10. Round 4: the flat-arena compiled VM, AES ILP, and the corrected workload

The reference workload moves to **N = 256, S = 6·256 = 1536** (the average of
the application's x side, S = 5·256, and y side, S = 7·256); the full
application costs 2× the single-workload numbers below. The aspirational
floor, counting CCRH invocations only at 5 ns each, is **12.5 ms per party**
per single workload (25 ms each for the doubled application).

### What landed

* **Flat-arena VM** (`comp_gc/arena.rs`): masks/labels for the chunk and
  extract phases live in dense typed arenas — `Vec<[u64; 2]>` for Z₂,
  `Vec<[u32; 128]>` for lane wires — addressed through a per-shape
  `WireLayout`. No `Label`, no `Option`, no allocation anywhere in phase
  execution; hash outputs expand directly into arena slots (k = 8/16 byte
  fast paths). Conversions to `Label` happen only at carry boundaries.
* **Compiled garbling** (`compile_garble`/`garble_compiled`): the recorded
  garble tape compiles once per shape into a typed instruction stream,
  validated at compile time exactly as a checked replay (wire membership +
  definedness simulation). The 78 repeat phases garble with *zero* System
  access except dynamic `Mul`-scalar loads — the r_i coefficients legally
  vary across same-shape phases, and baking them in produced wrong masks,
  caught immediately by the in-pipeline differential (`DFB_DIFF=1` verifies
  compiled ≡ reference replay on every keyed phase; the full suite passes
  with it enabled).
* **AES ILP** (`crypto`): 4-wide round-major CCRND keeps both M1 AES pipes
  fed — `expand` goes ~4.1 → ~1.75 ns/block, within ~10 % of the
  2-pipe hardware floor (10 fused rounds / 2 pipes ≈ 1.56 ns). Byte-identical
  (golden vectors + an old-loop oracle differential). `expand4` batches
  independent single-block hashes for future kernel use.
* **Def-4 nonce freshness**: the paper (App. A, Def. 4) requires that no two
  CCRH queries share a *nonce* — stricter than the (seed, nonce)-pair comment
  the crate carried. Solo-domain switch hashes now draw from per-phase
  windows allocated by the pipeline (`solo_nonce_next`), so gate ids reused
  across phases no longer reuse nonces. Bulk domain: in-System switch-group
  ids own [0, 2^32), kernels draw above `KERNEL_NONCE_FLOOR = 2^32`, with a
  hard bound below bit 63.
* **Smudging** (paper Thm. 5.2) documented as caller responsibility on
  `build_s_aff_streaming`: `b' = b + μ·p` before deriving residues when the
  evaluator will CRT-reconstruct over Z_M.
* Misc: System buffer pooling across phases; `Exec`'s internal modulus check
  demoted to debug (public `set` keeps its hard assert); `Add(x, x)`
  aliasing handled in arena ops.

### Where it stands (quiet machine, full clock, single workload S = 1536)

| | stream | garbler | evaluator | instructions | cycles |
|---|---|---|---|---|---|
| round 3 end | 0.12 s | ~50 ms | ~60 ms | 1.87 G | 0.38 G |
| **round 4** | **0.09 s** | **~35 ms** | **~53 ms** | **1.48 G** | **0.29 G** |

Doubled application: ~70 ms garbler + ~106 ms evaluator. Versus the
hash-only floor (12.5 ms/party single): the garbler is ~2.8× above, the
evaluator ~4× above. Cumulative since the original implementation: the
S=1280 stream went 2.33 s → 0.09 s at S=1536 — **~26× at matched clocks**.

### The measured gap to "physical"

Per party, single workload, all numbers measured or derived from counters:

| component | cost | reducible? |
|---|---|---|
| CCRH/AES (3.0 M λ-blocks × 1.75 ns) | ~5.3 ms | ~at hardware floor |
| body-kernel MACs (20.6 M slot×member pairs, NEON ÷4) | ~6 ms | 8-wide NEON ≈ −2 ms |
| extract lane math + hash-output unpacking | ~8 ms | NEON unpack ≈ −2 ms |
| eval-side schedule interpretation + journal discovery (`Exec`) | ~13 ms (evaluator only) | the last interpreter standing |
| build + boundaries + misc | ~4 ms | pooled already |

The protocol's compute is **not** hash-only: the IT-GC body alone does one
multiply-accumulate per (slot, member) pair — 20.6 M of them per party —
and the comp-GC's λ = 128-lane ring ops are real work the hash count never
sees. A hash-only floor estimate therefore undercounts by ~2–3×. What
remains structurally addressable: hand-rolled NEON for the generic-k unpack
and an 8-wide MAC kernel (~4 ms/party combined), and replacing `Exec`'s
worklist with a direct cleartext evaluator for the extract shape (~8 ms,
evaluator only). Beyond that, single-core gains require protocol changes;
across parties/workloads the work is embarrassingly parallel.
