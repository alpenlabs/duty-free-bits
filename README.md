# Switch System

Rust implementation of the switch system construction from [Duty-Free Bits](https://eprint.iacr.org/2026/476) (DFB), which projectivizes affine garbling schemes over prime fields at cost proportional to input + output labels.

The core construction is a switch system that computes affine maps **a**x + **b** over a CRT-friendly integer ring.

## Status

## Build

```sh
cargo build
cargo test
```

## References

- [KBH26] Khambhati, Bhattacharya, Heath. [Duty-Free Bits](https://eprint.iacr.org/2026/476). 2026.
- [Hea24] Heath. [Efficient Arithmetic in Garbled Circuits](https://eprint.iacr.org/2024/139). Eurocrypt 2024.

## Contributing

Contributions are generally welcome.
If you intend to make larger changes please discuss them in an issue
before opening a PR to avoid duplicate work and architectural mismatches.

For more information please see [`CONTRIBUTING.md`](/CONTRIBUTING.md).

## License

This work is dual-licensed under MIT and Apache 2.0.
You can choose between one of them if you use this work.
