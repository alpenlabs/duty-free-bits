# Artifact Evaluation — Duty-Free Bits

Roadmap for reproducing the claims of [Duty-Free Bits: Projectivizing Garbling
Schemes](https://eprint.iacr.org/2026/476) (Khambhati, Bhattacharya, Heath;
CCS '26). This file sits on `main`, at the commit tagged
[`ccs26-artifact`](https://github.com/alpenlabs/duty-free-bits/tree/ccs26-artifact) —
the paper's exact code (the rejection-sampling construction, [PR
#28](https://github.com/alpenlabs/duty-free-bits/pull/28), previously frozen
as tag `camera-ready-ccs26`) plus [PR
#29](https://github.com/alpenlabs/duty-free-bits/pull/29), an additive
reusability feature (`src/pgs.rs`, see below) that changes no benchmark
number. Every number in Table 1 and Figure 1 is unaffected by anything past
`camera-ready-ccs26`: the two commits between it and here are a defensive
edge-case guard unreachable from any driver (`2e1ad6a`), a formatting fix
(`5303118`), and this artifact's own `pgs.rs` addition.

See [`README.md`](README.md) for what the code does and
[`docs/architecture.md`](docs/architecture.md) for the construction. This
file is only about reproducing the paper's measurements.

## Requirements

- **Hardware.** None. The paper's absolute wall-clock numbers are from a
  single-threaded Apple M1 (2020 MacBook Pro); communication and hash counts
  are deterministic and machine-independent, and time *ratios* reproduce on
  any modern CPU (we cross-checked on an x86_64 machine while preparing this
  file: hash counts matched to 6 significant figures, wall-clock ratios
  matched to within 10%). NEON acceleration activates automatically on
  aarch64; there is a portable fallback for everything else, and both are
  tested bit-identical (`test_body_batch_simd_matches_scalar_differential`,
  `test_unpack_even_k_neon_matches_generic`).
- **Software.** Stable Rust, edition 2024 (≥ 1.85; tested with 1.90.0). No
  external system dependencies — `cargo build` fetches `mimalloc` and `rand`
  from crates.io.
- **Time.** The basic test (below) is seconds. Every individual benchmark in
  Experiment E1–E4 is under a minute; the full S-sweep behind Figure 1
  (Experiment E5) is under 5 minutes total.

## Basic test

```sh
cargo test --release
```

Expect `test result: ok. 100 passed; 0 failed; 6 ignored` in a few hundred
milliseconds. The 6 ignored tests are the benchmarks below, which are
`#[ignore]`d because they're meant to be run individually with specific
parameters, not as part of the suite.

## Major claims and how to check them

All benchmarks are single-threaded, release-mode, `#[ignore]`d tests, run
with `cargo test -r --lib <name> -- --ignored --nocapture`. Parameters are
environment variables; every command below is complete and copy-pasteable.
`N` is the input bit-length (paper fixes `N=256`) and `S` is the total affine
output dimension (the number of `a·x+b` maps computed in one call).

### C1 — Table 1: communication, hash counts, and wall-clock at two workload sizes

The paper reports these at `S=3,072` (the Embryo workload) and `S=153,600`
(50× larger). Both are one command:

```sh
N=256 S=3072   cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
N=256 S=153600 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
```

Each prints a table with garble/eval time (ms), hash counts (M), and
communication (MB/MiB) for both "one-hot CRT" (ours) and "bit-decomposition"
(the baseline). Communication and hash counts are exact and will match the
paper (0.34 MB / 6.9M / 6.9M at S=3,072; 11.5 MB / 286.7M / 285.2M at
S=153,600 — Table 1's `ours` columns exactly). Wall-clock will differ from
the paper's M1 numbers by whatever your CPU differs by; the *ratios*
(≈2× garbler, ≈4–5× evaluator) should hold regardless of hardware.

`ITERS` (default 30) and `WARMUP` (default 5) control the median-of-N
timing; lower them (e.g. `ITERS=5 WARMUP=2`) for a faster, noisier read at
`S=153,600`.

### C2 — Figure 1 (middle): exact hash-count split, cross-checked against the analytic ledger

```sh
N=256 S=1536 cargo test -r --features count-hashes --lib bench_axb_hashcounts -- --ignored --nocapture
```

Requires the `count-hashes` feature (a zero-overhead global counter, off by
default). Prints per-party CCRH block counts for both approaches and
compares the one-hot CRT measurement against the closed-form ledger
(`hash_count_cf=1,207,710` fixed + `hash_count_ncf` growing at
≈1,858.7/element) — the paper's "the hash-optimized ledger is cross-checked
to ±3% by `bench_axb_hashcounts`" claim (`docs/architecture.md` §5).

### C3 — Figure 1 (right) / Table 1 E2E rows: end-to-end latency beats the baseline despite slower per-party compute

```sh
N=256 S=3072   BW_MBPS=100  cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=3072   BW_MBPS=1000 cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=153600 BW_MBPS=100  cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=153600 BW_MBPS=1000 cargo test -r --lib bench_axb_network -- --ignored --nocapture
```

Models transmission as `bytes·8/bandwidth` (pure serialization delay, no
RTT) and reports garble + transmit + eval. Expect end-to-end wins of
≈22–28× at 100 Mbps and ≈3–5× at 1 Gbps (paper §8 "End-to-end latency"),
narrowing at 1 Gbps because the baseline's transmission cost shrinks enough
for its cheaper per-party compute to matter more. We reran the S=3,072 /
100 Mbps point while preparing this artifact and got 90.2 ms vs. the paper's
94 ms — a 22.7× win, matching the paper's 22×.

### C4 — Figure 1 (left): communication scales as a fixed one-time cost plus 593 bits/element

Communication is deterministic, so a single point plus the closed form is a
complete check (rather than rerunning 20 points to eyeball a curve):

```sh
N=256 S=2560 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
```

read the printed one-hot-CRT communication figure and compare against
`comm(S) = 112.2 KB + 74.14·S B` (paper: the CF cost is a constant ≈110 KiB,
the NCF cost is exactly `Σ⌈lg p_i⌉·S = 593·S` bits at every point). Hashes
follow the same shape, `hashes(S) = 1,207,710 + 1,858.7·S` (per-party CCRH
calls, rejection-sampling retries included — this is the formula behind
Table 1's `ours` hash columns, which we confirmed by direct measurement at
both its endpoints in C1: 6.92M at S=3,072, 286.7M at S=153,600, both exact
matches). The paper's full sweep (`S ∈ {1536, 3072, 6144, 12288, 24576,
49152, 98304, 153600}`, all reproducible the same way as C1) is:

| S | ours comm | baseline comm (GRR) | ours hashes | baseline hashes G/E |
|---|---|---|---|---|
| 1,536 | 0.22 MB | 12.0 MiB | 4.06M | 1.57M / 0.79M |
| 3,072 | 0.34 MB | 24.1 MiB | 6.92M | 3.15M / 1.57M |
| 6,144 | 0.57 MB | 48.2 MiB | 12.63M | 6.29M / 3.15M |
| 12,288 | 1.02 MB | 96.4 MiB | 24.04M | 12.58M / 6.29M |
| 24,576 | 1.93 MB | 192.8 MiB | 46.88M | 25.17M / 12.58M |
| 49,152 | 3.75 MB | 385.5 MiB | 92.55M | 50.33M / 25.17M |
| 98,304 | 7.40 MB | 771.0 MiB | 183.90M | 100.66M / 50.33M |
| 153,600 | 11.50 MB | 1,204.7 MiB | 286.70M | 157.29M / 78.64M |

Communication and baseline figures are exact closed forms, unaffected by the
rejection-sampling randomness (bit-decomposition has no rejection step; the
one-hot CRT side's rejection sampling changes only local hash calls, never
transmitted bits), so those columns are exact at every row. The "ours
hashes" column is the expected value of a formula with a small
input-independent random component (which windows get rejected); individual
runs land within a fraction of a percent of it, as in C1.

### C5 — Embryo and Argo MAC application totals (§7, Appendix B): 51× and 23× improvement over prior work

These are *derived*, not single-command, quantities — same as in the paper
(§8, Appendix B): the fixed CRT-conversion cost is paid once per input
coordinate, so a 2-coordinate application (Embryo has `x, y`; Argo MAC's
encoding is likewise 2 coordinates) pays it twice, while the S-dependent
term uses the *combined* output dimension in one `bench_axb_comparison` run.

```sh
N=256 S=3072 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture   # Embryo: Sx=5·256, Sy=7·256
N=256 S=1092 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture   # Argo MAC: 12·κ, κ=91
```

Embryo total ≈ (printed comm at S=3,072, ≈0.34 MB) + one more 110 KiB fixed
cost (the second coordinate) + a ≈0.4–0.5 KiB curve-check term ≈ 442 KiB,
against BABE's reported 22.16 MiB — a 51× improvement. Argo MAC total ≈
(printed comm at S=1,092) + one more 110 KiB fixed cost ≈ 298 KiB, against
[18]'s reported 6.8 MiB — a 23× improvement. Both ratios and the ≈110 KiB /
593 bit-per-element constants are also checked directly against the
closed-form analysis in Appendix B, independent of any benchmark run.

### Bonus: per-stage attribution (not a paper table, but backs §8's cost breakdown prose)

```sh
N=256 S=1536 cargo test -r --lib bench_axb_stages -- --ignored --nocapture
```

Splits `build_s_aff` into chunk/extract/fold/body wall-clock, backing the
"body is MAC-bound at scale" / "chunk+extract+fold sit near their AES-block
floor" claims in `docs/architecture.md` §5.

## Correctness (beyond the benchmarks)

`cargo test --release` (part of the basic test above) also exercises:
end-to-end known-answer tests against a direct CRT arithmetic oracle
(`test_s_aff_sweep`, `test_s_aff_edge_regimes`), the carry invariant
`label = mask + value·Δ` at every wire (`test_*_label_mask_invariant`), and
NEON-vs-scalar differential tests for both the CCRH core and the body/cast
SIMD kernels. `docs/architecture.md` §6 lists the full coverage story.

## Reusability: the construction as a composable API (`src/pgs.rs`)

Not a paper claim, so not one of C1–C5, but directly relevant to the
*Artifacts Evaluated—Reusable* bar: `build_s_aff` (what the benchmarks call)
runs garbler and evaluator in one pass. `src/pgs.rs` exposes the same
protocol — identical steps, nonce layout, and ciphertexts — split into
`garble`/`encode`/`eval`, so a caller can compose this affine-map scheme as
a sub-protocol inside a larger garbled circuit (an outer scheme whose input
labels are this protocol's outputs) instead of only running it standalone.
`pgs::tests::test_pgs_split_matches_build_s_aff` proves the split API is the
same protocol: from the same generator seed, it and `build_s_aff` emit
identical residues.

```sh
cargo test --release pgs::
```

reproduces this directly (2 tests, part of the 100 in the basic test above).

## Known limitations

Single-threaded, research code (`docs/architecture.md` "Status"). The
codebase establishes correctness and cost, not machine-checked security — a
statement `docs/architecture.md` §4 makes explicitly ("What this codebase
establishes is correctness and cost, not a security reduction"); the
security proof is in the paper.
