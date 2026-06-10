# Cost profile: `build_s_aff_streaming` at production scale (N=256, S=1280)

*2026-06-09, Apple M1 (P-core: 128 KB L1d, 12 MB shared L2, ~3.2 GHz), single-threaded,
`cargo test --release`, mimalloc.*

This documents where the 1 s garble / 1 s eval actually went, the experiments
that localized each cost, the optimizations applied (with measured effect), and
the remaining gap to the genuine instruction floor.

**Headline: total wall time went 2.33 s → 0.51 s (4.6×) with no protocol or
wire-format changes. Hashing was never the bottleneck; label bookkeeping was.**

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
