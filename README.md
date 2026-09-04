# Duty-Free Bits

Rust implementation of the one-hot CRT projectivization from [Duty-Free Bits](https://eprint.iacr.org/2026/476) [KBH26, to appear at ACM CCS 2026]: garbled evaluation of affine maps **a**x + **b** over a CRT primorial ring as a single straight-line protocol, plus a bit-decomposition baseline the benchmarks compare against.

## Status

Complete garble + evaluate pipeline (chunk → extract → fold → body), single-threaded, research code — see [`docs/architecture.md`](docs/architecture.md) for the construction, storage layout, cryptography, and testing story.

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

## References

- [KBH26] Khambhati, Bhattacharya, Heath. [Duty-Free Bits](https://eprint.iacr.org/2026/476). ACM CCS 2026.
- [Hea24] Heath. [Efficient Arithmetic in Garbled Circuits](https://eprint.iacr.org/2024/139). Eurocrypt 2024.

## Contributing

Contributions are generally welcome.
If you intend to make larger changes please discuss them in an issue
before opening a PR to avoid duplicate work and architectural mismatches.

For more information please see [`CONTRIBUTING.md`](/CONTRIBUTING.md).

## License

This work is dual-licensed under MIT and Apache 2.0.
You can choose between one of them if you use this work.
