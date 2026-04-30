# Switch System

Reference Rust implementation of a switch system construction for affine maps over large prime fields.

## Reproducing the cost simulation

The communication and hash-count simulated numbers are produced by a single test. To reproduce:

```sh
cargo test --release test_s_aff_80_metrics -- --ignored --nocapture
```

The test builds the n=256, 80-prime affine switch system independently for each S in {1, 256, 512, ..., 256·20}, reads the `join_complexity_{cf,ncf}` and `hash_count_{cf,ncf}` counters from the `System` struct, and prints a table of communication (in bits and KiB) and hash invocations per S. 
The S=1 build also runs the constructed circuit through the `Exec` engine and verifies the output residues against a CRT reconstruction, providing end-to-end correctness.

Wall time: ~7 minutes; peak memory: ~6.7 GB at S=5120.

## Standard tests

```sh
cargo test --release
```

Runs 124 unit and integration tests. Should complete in under 2 seconds.
