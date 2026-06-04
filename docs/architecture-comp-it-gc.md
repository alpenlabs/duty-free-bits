# Architecture: Computational GC ∘ Information-Theoretic GC (v2)

Status: **design, plan-reviewed** (branch `feat/comp-it-gc`). Incorporates the
systems / cryptography / software / audit reviews of v1. No behavior change vs
`feat/garble-eval`; this restructures the code so it mirrors the protocol.

## 0. Guiding principle

The construction is a **composition of two garbling schemes** joined by one CCRH
bridge. The code should make that the top-level structure:

```
   bin(x) ─[ computational GC ]→ CF labels on OHE-CRT(x) ─[ CCRH bridge ]→ NCF pads ─[ IT GC ]→ {a·x+b mod p_i}
            (Yao-style, λ-fold,                (the binary one-hot h_p)        (information-theoretic,
             Δ-correlation, CCRH)                                               one-time-pad, no λ blowup)
```

This is the paper's CF/NCF wire distinction (Def. 2) read as a module boundary.
The paper formalizes **one** switch system with per-wire CF/NCF tags; the split is
the right *engineering* decomposition **of this construction** (Phase 1 needs the
general bidirectional engine; Phase 3 is direct linear algebra over one-time pads).
It must stay *provably equivalent* to the single-framework model — enforced by the
existing general gate engine acting as a reference oracle (§6).

## 1. Paper grounding (checked against `Duty Free Bits (4).pdf`)

- **CF vs NCF is a wire type, declared structurally** (Def. 2, §6.1). CF = product
  of `Z_{2^k}` rings, length-λ labels `X + x·Δ_R`, the only legal switch *controls*.
  NCF = any finite group, single group element `X + x` (one-time pad), **no λ blowup**.
- **Join width** (Def. 3): `λ·lg|G|` if both CF, else `lg|G|` (= the garbled program).
- **Nonce legality** (Def. 4, App. A): `circ(Δ,x,i,b)=H(x⊕Δ,i)+bΔ`; legal iff **no two
  queries reuse a `(key, nonce)`**. Every switch uses unique gate id as nonce.
- **Garbler/Evaluator asymmetry** (App. A): Δ appears **only** in the switch output —
  `Y=H(S⊕Δ,gid)+X+(y−x)Δ` (CF) or `+(y−x)` (NCF). Joins: garbler appends `X−Y`,
  evaluator subtracts. Homomorphism/subgroup/pair gates are symmetric `φ`.
- **Zero switch communication** (§3.3): every switch control is inferrable from the
  evaluator's cleartext `x` (in BOTH phases), so switches reveal nothing — **no
  point-and-permute LSB**. The garbled program is exactly the join material (+ output
  masks for final decode). *The current general engine emits one redundant ctrl LSB per
  switch (`program.rs:48`); the comp_gc rework removes it by deriving controls from `x`,
  exactly as `Exec` does — bringing communication to the paper's join-width.*
- **S_aff stages** (§6.3, App. B): Phase 1 `bin-to-word` (Lemma 6.3, all CF) → free CRT
  accumulation (homomorphism gates, no join width) → sub-chunk extract + fold to a
  length-`p_i` **binary OHE `h_p`** (App. B, CF) → Phase 3 `scale-hot` (Lemma 6.2) +
  `hot-to-ring` (Lemma 6.1), the **NCF** readout of `a·x+b`. Boundary = `h_p`.
- **Verified by the crypto review:** the kernel's body switch (`o=H(h_p[k],nonce)`,
  constant data `z=0`) is an exact instance of `scale-hot` with App. A's NCF switch
  (`X=0, y−x=0` ⇒ no Δ term) + join (`X−Y`); Δ enters only through the `h_p` control
  seed (non-hot agree, hot recovered by the join). **The decomposition is faithful.**

## 2. The boundary contract

Per CRT prime `p_i`, comp_gc hands it_gc the garbled binary one-hot `h_p` of
`x mod p_i` — **both** the garbler's masks **and** the evaluator's labels (they cross
together), plus the cleartext active position for the evaluator:

```
comp_gc output:  h_p carry = (masks: [CfLabel; p_i], labels: [CfLabel; p_i])   // a slice of CarryItem
it_gc   input :  h_p carry  +  (a_i, b_i): [u64; S]  +  hot = x mod p_i  +  nonce_base
it_gc   output:  [NcfShare; S]                                                 // NCF Z_{p_i} of a_{i,j}·x+b_{i,j}
```

- The carry stays `pipeline::CarryItem` (no new type); the boundary is `it_gc::body::{garble,eval}`
  consuming the mask/label halves of the prime's `h_p` carry slice.
- The bridge is `pad = H(h_p[k], nonce)` reinterpreted as `NcfShare` (the NCF scale-hot
  switch with constant data `z=0`, realizing `y_k := 0 ⇐ h_p[k]=0`).
- **`hot` and `nonce_base` are passed by value into it_gc** — it_gc never borrows the
  shared allocator and never depends on `comp_gc`/`system`.

## 3. Module structure

```
src/
  ring/            label/share arithmetic, no protocol logic
    cf.rs            CfLabel (λ-fold Z_{2^k}) + add/sub/scalar_mul/mod2k/div2k + Δ_R + label↔bytes
    ncf.rs           NcfShare(rep:u64, modulus:u64) + mod ops + neg(const) + extract_ncf; debug_assert(p ≤ 409)
  cost.rs          shared Cost{ join_cf, join_ncf, hash_cf, hash_ncf } — pure fold; both schemes accumulate
  crypto/          the CCRH primitive, Label-free
    ccrh.rs          trait Ccrh { fn hash(&self, seed:&[u8;16], nonce:u64, out:&mut [u8]); }  + hash_bulk into &mut[u8]
    nonce.rs         NonceAllocator: domain-tagged, per-scheme disjoint sub-spaces; debug-asserts Def.4 legality
    aarch64/         Neon AES-NI backend (4-block interleaved), behind Ccrh, #[inline]
    portable.rs      constant-time software fallback so the crate compiles off-aarch64   [Gate 0 stub, R-final real]
  comp_gc/         computational Yao GC over CF wires (Phase 1)
    system.rs        Circuit DAG + SwitchGroup (groups CF-controlled switches w/ NCF outputs); folds into Cost
    garbler.rs       mask propagation + Δ + emit (switch LSB, join diff); ALWAYS emits ctrl LSB (invariant)
    evaluator.rs     label propagation + ctrl-branch + consume; shares affine arms w/ garbler via one helper
    builders.rs      bin_to_word, sub_chunk_extract, fold_to_mod_ohe  (bin(x) → h_p)
  it_gc/           information-theoretic GC over NCF shares (Phase 3) — NO dep on comp_gc/system
    body.rs          scale-hot + hot-to-ring, garbler & evaluator; takes hot+nonce_base by value (was affine_kernel)
  crt/             crt.rs, bigint.rs (leaf)
  pipeline/        Pipeline + CarryItem — the real composition point; owns the NonceAllocator (hands out bases by value)
  affine.rs        thin schedule on Pipeline; build_s_aff_streaming; build_s_aff/garble/eval stay PUBLIC
  reference/       (test-only) Exec cleartext oracle [system level] + Phase-3-as-NCF-gates oracle [pipeline level]
  tests.rs
```

Dependency DAG (no upward edges): `ring → {cost, crypto} → {comp_gc, it_gc} → pipeline → affine`;
`reference`/`tests` on top. Acceptance: builds warning-clean under
`unused_crate_dependencies="deny"` on **both** aarch64 and a non-aarch64 target
(backend deps under `[target.'cfg(...)'.dependencies]`).

## 4. Core types & invariants

- `ring::ncf::NcfShare{rep,modulus}` — modular `u64` (the kernel's `mod_*`, `#[inline(always)]`).
  `debug_assert(modulus ≤ 409)` at construction — the concrete-params bound the u64 fast paths
  (`a*b` in `mod_mul`, compare-subtract in add) actually rely on, not merely `≤ 2^32`.
- `crypto::Ccrh` — **`Label`-free**, `(seed:&[u8;16], nonce:u64, out:&mut[u8])`. Production path is
  **generic-monomorphized over the single concrete backend** (or a cfg-selected free fn) — **never `&dyn`**
  (vtable on the innermost loop regresses at `opt-level`). Trait methods `#[inline]`; R1 has a `cargo asm`
  acceptance gate proving the AES round loop still inlines into `hash`.
- `crypto::NonceAllocator` — **domain-tagged**: comp_gc (`gid`/header) and it_gc (body) draw from
  **provably disjoint** sub-spaces (lifting today's implicit `hash_solo`/`hash_bulk` bit-63 split into the
  contract). API `next_window(scheme, count) -> base`, drawn **once per phase/batch**; the kernel derives
  position nonces as `base + i`. The disjointness/Def.4 legality check is `cfg(debug_assertions)`-only; the
  release allocator is a plain counter (no `Vec`). Per-prime windows are **precomputed by prefix-sum** so the
  prime loop is deterministic and embarrassingly parallel.
- Mask vs label: distinguished by **which side's function produces them** (garbler→masks, evaluator→labels).
  We do **not** add `Mask`/`Label` newtypes (acceptable for 7k LoC) — and we do **not** claim a type-level
  safety we aren't buying; `CarryItem` legitimately holds both.
- **No `RingElem` trait** spanning both modules (would re-couple). The only shared arithmetic is the Exec
  oracle's `Val` ops, which stay concrete and **test-only** in `reference/`.

## 5. What stays NOT unified (honest seams)

- **Garbler vs evaluator** stay two functions *within each module* (App. A asymmetry). Only Switch/Join differ
  by role; the affine arms (Add/Sub/Mul/Mod2k/Div2k = homomorphism/subgroup gates) are shared by one helper.
- **comp_gc and it_gc** are separate schemes (computational vs IT), not one engine with a flag.
- **The evaluator derives ALL switch controls from its cleartext `x`, in BOTH phases** — this is a uniform
  global property, NOT a comp_gc/it_gc asymmetry. (Corrects v1/v2's mistaken "it_gc-only" framing.) Neither
  scheme uses point-and-permute or emits a switch LSB; both evaluators run cleartext control propagation (like
  `Exec`) alongside label propagation. The only garbled communication is join widths (+ output decode). The
  comp_gc evaluator therefore needs the cleartext phase inputs (threaded from `x`), not just labels — a real
  change from today's `eval_with_labels`, validated against `Exec`.

## 6. Correctness / test strategy (the safety net — landed FIRST, against current code)

1. **Reference oracle = the existing general gate engine** (System + garbler + evaluator) expressing the *full*
   S_aff (Phase 3 as NCF gates) + `Exec` cleartext. Not new code — `build_s_aff`/`garble`/`eval` already do this.
2. **Three-way differential** (property test, swept over prime ∈ first-80, **S across the 128 batch boundary
   {1,127,128,129,256,257}**, every hot position, several Δ): `Exec value == (comp_gc∘it_gc) decode ==
   gate-oracle garble∘eval decode`, run against the **real** CCRH (not the zero-stub).
3. **it_gc vs gate-oracle Phase 3 — mask-for-mask and label-for-label** (distributional, not value-equality):
   a value test passes even under a two-time-pad break, so this must compare the actual garbled material.
4. **Nonce legality (permanent)**: `NonceAllocator` debug-asserts disjoint windows; the two-time-pad regression
   is generalized to **≥3 consecutive batches × multiple primes** and kept forever.
5. **Golden vectors** for `Ccrh`/`hash_bulk`/`extract_ncf` pinning the R1 trait boundary byte-for-byte.
6. **Cost = one fold**: `it_gc` batch cost and comp_gc circuit cost both fold into the shared `Cost`; assert the
   kernel's per-batch cost equals what the gate-oracle reports for the same sub-circuit (kills the
   `RESIDUE_BATCH_SIZE == λ` coincidence; introduce distinct named constants for the three 128s).
7. **Production parity** (kept green throughout): `test_streaming_s_aff_matches_all_at_once` + the `N=256 S=1280`
   scale run — same **outputs**. Add a **CI-runnable multi-batch variant** (e.g. S≈300) since the current
   equivalence fixture (S=3) never crosses the 128 boundary and the scale test is `#[ignore]`d.
8. **Zero switch communication** (paper §3.3): after the comp_gc rework (controls from `x`), assert the garbled
   program contains **no** switch LSBs — communication = join width (+ output masks). This is a behavior change
   (program shrinks ~60% at scale) and a paper-faithfulness check; pre-rework the general engine emits one
   redundant LSB/switch, so this assertion lands *with* the rework, not before it.

## 7. Migration sequence (reviewer-corrected: net + build FIRST, then stages)

**GATE 0 — blocking, before any refactor commit:**
- G0.1 Fix the off-aarch64 compile break (cfg-gate `hash.rs` aarch64 imports + a stub/portable `Ccrh`) so the
  crate builds on non-aarch64. Add a CI lane that builds+tests on the CCRH target arch (aarch64), or document
  that security tests run locally only. *(Source files already committed on this branch.)*
- G0.2 `opt-level="z"` → `3` as a standalone change; re-baseline the ~5.5s streaming run (z vs 3, + peak RSS).

**GATE 1 — land the safety net against the CURRENT code and prove it green pre-refactor:**
- G1 = test-strategy items 2–7 above. A net written after a move can't tell "refactor correct" from "oracle
  written to match new code."

**Then the stages (each compiles + all tests green; each its own reviewable step):**
- **R1 — crypto**: `Ccrh` trait (Label-free, `&mut[u8]`, monomorphized, `#[inline]`, inlining acceptance gate) +
  4-block-interleaved AES (so R1 is "behavior-identical, faster") + `NonceAllocator` (domain-tagged, by-value
  windows, debug-only legality) + reusable per-(prime/thread) hash scratch buffer.
- **R2 — ring**: extract `ring::cf`/`ring::ncf`; `NcfShare` newtype + `p ≤ 409` debug_assert; move `extract_ncf`
  into `ring::ncf`. Behavior-identical.
- **R3 — it_gc**: move the kernel to `it_gc/body.rs`, boundary-contract API, nonce windows by value, **explicit
  non-goal: no dependency on comp_gc/system**. Add the it_gc-vs-oracle mask/label differential. Behavior-identical.
- **R4 — comp_gc**: move System+garbler+evaluator+builders; extract shared `Cost`; share the garbler/evaluator
  affine arms via one helper. **Eliminate point-and-permute**: garbler emits no switch LSB; the evaluator derives
  every control from cleartext `x` (cleartext propagation in lockstep with labels). Validate against `Exec` +
  assert program = join width (item 8). `reference/` keeps the NCF-gate oracle + Exec.
- **R5 — pipeline/affine**: rewire streaming as comp_gc headers → it_gc bodies; **keep `build_s_aff`/`garble`/`eval`
  as public API** (not demoted to oracle). Design the mask+label boundary carry explicitly. Add the three-way sweep.
- **R6 — parallelism**: per-prime nonce windows + `Send` per-prime state + `par_iter`/`scope` over the 80 primes,
  telemetry folded post-join (needs `rayon` or `std::thread::scope`). Note peak RSS becomes `T × phase + carry`.

**DROPPED / DEFERRED:** the v1 `Gate`→`AffineGate`/`GarbledGate` split (cosmetic, ripples through three exhaustive
matches) — instead fix the `Wire{wid:0}` sentinel with `Option<Wire>`. `Result`-at-entry-points: only for genuine
input-validation failures, low priority. Constant-time portable `Ccrh`: it is a *new crypto implementation*, must
meet R1's golden-vector + differential bar (not "polish").

## 8. Scope note

The motivating security fire (nonce reuse) is already out (fixed on the parent branch). R1–R2 are
correctness/portability hardening worth doing regardless. R3–R6 are the readability/parallelism win the user has
chosen; they are sequenced behind a proven test net so their correctness risk is bounded, and the all-at-once
public path + reference oracle keep a differential anchor at every step.
