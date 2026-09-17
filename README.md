# Duty-Free Bits

Rust implementation of the one-hot CRT projectivization from [Duty-Free Bits](https://eprint.iacr.org/2026/476) [KBH26, to appear at ACM CCS 2026]: garbled evaluation of affine maps **a**x + **b** over a CRT primorial ring as a single straight-line protocol, plus a bit-decomposition baseline the benchmarks compare against.

## Requirements

- **Hardware.** None. The paper's wall-clock numbers are from a
single-threaded Apple M1 (2020 MacBook Pro). NEON
acceleration activates automatically on aarch64; there is a portable
fallback for everything else.
- **Software.** Stable Rust, edition 2024 (≥ 1.85; tested with 1.90.0).

## Build

```sh
cargo build
cargo test
```

## Benchmarks

All benchmarks are `#[ignore]`d tests, run manually in release mode (env overrides in parentheses):

- `cargo test -r --lib bench_axb_comparison -- --ignored --nocapture` (env: N, S, ITERS, WARMUP) — field-to-field head-to-head vs the bit-decomposition baseline.
- `BW_MBPS=100 cargo test -r --lib bench_axb_network -- --ignored --nocapture` — end-to-end latency over a simulated bandwidth-limited link.
- `cargo test -r --features count-hashes --lib bench_axb_hashcounts -- --ignored --nocapture` — per-party CCRH block counts, cross-checked against the analytic ledger.
- `cargo test -r --lib bench_axb_stages -- --ignored --nocapture` — per-stage wall-clock split of `build_s_aff`.

## Reproducing the paper's results

`N` is the input bit-length (paper fixes `N=256`) and `S` is the total
affine output dimension (the number of `a·x+b` maps computed in one call).

### Table 1: communication, hash counts, and wall-clock at two workload sizes

The paper reports these at `S=3,072` (the Embryo workload) and `S=153,600`
(50× larger).

```sh
N=256 S=3072   cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
N=256 S=153600 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
```

Each prints a table with garble/eval time (ms), hash counts (M), and
communication (MB/MiB) for both "one-hot CRT" (ours) and "bit-decomposition"
(the baseline). Communication and hash counts should match the
paper (0.34 MB / 6.9M / 6.9M at S=3,072; 11.5 MB / 286.7M / 285.2M at
S=153,600 — Table 1's `ours` columns). Wall-clock may differ slightly from
the paper's M1 numbers based on CPU.

### Figure 1 (middle): exact hash-count split, cross-checked against the analytic ledger

```sh
N=256 S=1536 cargo test -r --features count-hashes --lib bench_axb_hashcounts -- --ignored --nocapture
```

Requires the `count-hashes` feature (a zero-overhead global counter, off by
default). Prints per-party CCRH block counts for both approaches.

### Figure 1 (right) / Table 1 E2E rows: end-to-end latency beats the baseline despite slower per-party compute

```sh
N=256 S=3072   BW_MBPS=100  cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=3072   BW_MBPS=1000 cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=153600 BW_MBPS=100  cargo test -r --lib bench_axb_network -- --ignored --nocapture
N=256 S=153600 BW_MBPS=1000 cargo test -r --lib bench_axb_network -- --ignored --nocapture
```

Models transmission as `bytes·8/bandwidth` (serialization delay, no
RTT) and reports garble + transmit + eval.

### Figure 1 (left): communication scales as a fixed one-time cost plus 593 bits/element

```sh
N=256 S=2560 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
```

The paper's full sweep (`S ∈ {1536, 3072, 6144, 12288, 24576, 49152, 98304, 153600}`, all reproducible the same way) is:


| S       | ours comm | baseline comm (GRR) | ours hashes | baseline hashes G/E |
| ------- | --------- | ------------------- | ----------- | ------------------- |
| 1,536   | 0.22 MB   | 12.0 MiB            | 4.06M       | 1.57M / 0.79M       |
| 3,072   | 0.34 MB   | 24.1 MiB            | 6.92M       | 3.15M / 1.57M       |
| 6,144   | 0.57 MB   | 48.2 MiB            | 12.63M      | 6.29M / 3.15M       |
| 12,288  | 1.02 MB   | 96.4 MiB            | 24.04M      | 12.58M / 6.29M      |
| 24,576  | 1.93 MB   | 192.8 MiB           | 46.88M      | 25.17M / 12.58M     |
| 49,152  | 3.75 MB   | 385.5 MiB           | 92.55M      | 50.33M / 25.17M     |
| 98,304  | 7.40 MB   | 771.0 MiB           | 183.90M     | 100.66M / 50.33M    |
| 153,600 | 11.50 MB  | 1,204.7 MiB         | 286.70M     | 157.29M / 78.64M    |


### Embryo and Argo MAC application totals (§7, Appendix B): 51× and 23× improvement over prior work

Derived via the following commands:

```sh
N=256 S=3072 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture   # Embryo: Sx=5·256, Sy=7·256
N=256 S=1092 cargo test -r --lib bench_axb_comparison -- --ignored --nocapture   # Argo MAC: 12·κ, κ=91
```

Embryo total ≈ (printed comm at S=3,072, ≈0.34 MB) + one more 110 KiB fixed
cost (the second coordinate) + a ≈0.4–0.5 KiB curve-check term ≈ 442 KiB,
against BABE's reported 22.16 MiB — a 51× improvement. Argo MAC total ≈
(printed comm at S=1,092) + one more 110 KiB fixed cost ≈ 298 KiB, against
[18]'s reported 6.8 MiB — a 23× improvement.

### Correctness

`cargo test` also performs: end-to-end known-answer tests against a direct
CRT arithmetic oracle.

## Reusability: the construction as a composable API (`src/pgs.rs`)

`build_s_aff` (what the benchmarks call) runs garbler and evaluator in one
pass. `src/pgs.rs` exposes the same protocol split into `garble`/`encode`/`eval`, so a caller
can compose this affine-map scheme as a sub-protocol inside a larger
garbled circuit (an outer scheme whose input labels are this protocol's
outputs) instead of only running it standalone.

## Known limitations

Single-threaded, research code.

## References

- [KBH26] Khambhati, Bhattacharya, Heath. [Duty-Free Bits](https://eprint.iacr.org/2026/476). ACM CCS 2026.
- [Hea24] Heath. [Efficient Arithmetic in Garbled Circuits](https://eprint.iacr.org/2024/139). Eurocrypt 2024.

## License

This work is dual-licensed under MIT and Apache 2.0.
You can choose between one of them if you use this work.