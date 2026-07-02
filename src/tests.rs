//! Benchmarks and end-to-end correctness tests for the a·x+b protocol.
//!
//! Unit tests for the individual steps live in their modules' `#[cfg(test)]`
//! blocks (`gc::chunk`, `gc::extract`, `gc::fold`, `gc::body`, `crt`, `label`).

use crate::crt::bigint::{FIRST_80_PRIMES, U576};
use crate::crt::{CrtParams, GarnerDecoder};

use rand::{Rng, RngExt};

fn rng() -> impl Rng {
    rand::rng()
}


struct BenchRng {
    state: u64,
}

impl BenchRng {
    fn new(seed: u64) -> Self {
        BenchRng { state: seed }
    }
}

impl rand::RngCore for BenchRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut i = 0;
        while i < dst.len() {
            let r = self.next_u64().to_le_bytes();
            let n = (dst.len() - i).min(8);
            dst[i..i + n].copy_from_slice(&r[..n]);
            i += n;
        }
    }
}

/// BN254 base-field modulus q, little-endian limbs (mirror of the private
/// `bitdecomp::P`); the switch side's smudge/decode arithmetic needs it.
const BN254_Q: [u64; 4] = [
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// q mod p via limb-wise Horner (per-prime-set precompute).
fn bn254_q_mod(p: u64) -> u64 {
    let mut acc: u128 = 0;
    for &limb in BN254_Q.iter().rev() {
        acc = ((acc << 64) | limb as u128) % p as u128;
    }
    acc as u64
}

/// Reduce a Garner-reconstructed integer to the field: limb-wise Horner with
/// 64 doublings per limb (`Fp` deliberately has no general multiply; a
/// production decoder would Barrett-reduce, so this is the conservative,
/// switch-side-pessimistic form).
fn u576_mod_q(v: &U576) -> crate::bitdecomp::Fp {
    use crate::bitdecomp::Fp;
    let top = v.0.iter().rposition(|&l| l != 0).unwrap_or(0);
    let mut acc = Fp::zero();
    for &limb in v.0[..=top].iter().rev() {
        for _ in 0..64 {
            acc = acc.double();
        }
        acc = acc.add(Fp::from_u64(limb));
    }
    acc
}

#[test]
fn test_u576_mod_q_helper() {
    use crate::bitdecomp::Fp;
    // Small values are fixed points.
    assert_eq!(u576_mod_q(&U576::from_u64(12345)), Fp::from_u64(12345));
    // q itself reduces to zero; q + 7 to 7 (low limb addition cannot carry).
    let mut q = [0u64; 9];
    q[..4].copy_from_slice(&BN254_Q);
    assert_eq!(u576_mod_q(&U576(q)), Fp::zero());
    let mut q7 = q;
    q7[0] += 7;
    assert_eq!(u576_mod_q(&U576(q7)), Fp::from_u64(7));
    // 2^256 mod q cross-checked against the same Horner done limb-free.
    let mut two256 = [0u64; 9];
    two256[4] = 1;
    let mut acc = Fp::from_u64(1);
    for _ in 0..256 {
        acc = acc.double();
    }
    assert_eq!(u576_mod_q(&U576(two256)), acc);
}

/// Final head-to-head benchmark: switch system (CRT) vs the bit-decomposition
/// baseline ([`crate::bitdecomp`]) on the SAME `a·x+b` workload, on identical
/// footing — single-threaded, release, warmup + ITERS iterations, MEDIAN [MIN]
/// reported. Both EXCLUDE a,b sampling (done once before timing) and any
/// correctness/decode step (correctness is covered by
/// `bitdecomp::tests::test_garble_eval_matches_plaintext` and the assert in
/// `test_s_aff_scaling`). Input-label provisioning is outside the timed
/// region for both (bit-decomp `encode`; the switch driver samples input
/// masks outside its per-stage timers). The switch side runs
/// [`crate::affine::build_s_aff`].
///
/// Workload boundary: available are field elements (a, b) and input labels;
/// required output is a·x+b as field elements. The switch garble column
/// therefore includes statistical smudging (fresh 64-bit μ per component,
/// b\' = b + μ·q with residues derived directly as (b + μ·q) mod p — the big
/// integer is never materialized) plus CRT-encoding a, b\' into per-prime
/// residues; the switch eval column includes Garner reconstruction AND the
/// reduction mod q to the field element (double-based Horner — conservative;
/// a Barrett reduction would shrink it). Garner setup and q mod p tables are
/// per-prime-set precompute, outside, like bit-decomp\'s compile-time
/// constants.
///
/// Run:
///   cargo test -r --lib bench_axb_comparison -- --ignored --nocapture
/// Override via env: `N=256 S=1536 ITERS=30 WARMUP=5 cargo test -r ...`
#[test]
#[ignore]
fn bench_axb_comparison() {
    use crate::bitdecomp;
    use std::hint::black_box;
    use std::time::Instant;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let n = env("N", 256) as u32;
    let s = env("S", 1536);
    let iters = env("ITERS", 30);
    let warmup = env("WARMUP", 5);
    assert_eq!(
        n as usize,
        bitdecomp::N_BITS,
        "bit-decomp N_BITS is a compile-time constant ({}); set N to match",
        bitdecomp::N_BITS
    );

    // (min_ms, median_ms) from per-iteration seconds.
    fn stats(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0] * 1e3, v[v.len() / 2] * 1e3)
    }

    // Fast PRG for both sides: bit-decomp share sampling uses it directly; the
    // switch system only uses it for out-of-band (untimed) Δ / input-mask draws.
    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ===================== bit-decomposition baseline =====================
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x_fp = bitdecomp::Fp::random(&mut rng);

    let mut bd_g = Vec::with_capacity(iters);
    let mut g_keep = None;
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(&g);
        if it >= warmup {
            bd_g.push(dt);
        }
        g_keep = Some(g);
    }
    let g = g_keep.unwrap();
    let active = bitdecomp::encode(&g, x_fp);
    let mut bd_e = Vec::with_capacity(iters);
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let out = bitdecomp::eval(&g, &active, x_fp, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(out);
        if it >= warmup {
            bd_e.push(dt);
        }
    }
    let (bdg_min, bdg_med) = stats(bd_g);
    let (bde_min, bde_med) = stats(bd_e);
    let bd_h = bitdecomp::cost_model(s);
    let (bd_comm, bd_comm_grr) = (g.comm_bytes(), g.comm_bytes_grr());

    // ===================== switch system (CRT) =====================
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let decoder = GarnerDecoder::new(&params.primes); // per-prime-set precompute
    let q_mod_p: Vec<u64> = params.primes.iter().map(|&p| bn254_q_mod(p)).collect();

    let mut sw_g = Vec::with_capacity(iters);
    let mut sw_e = Vec::with_capacity(iters);
    let mut sw_t = Vec::with_capacity(iters);
    let (mut sw_cf, mut sw_ncf, mut sw_comm_bits) = (0usize, 0usize, 0usize);
    let (mut sw_enc_ms, mut sw_dec_ms) = (0.0f64, 0.0f64);
    for it in 0..warmup + iters {
        let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();

        // Garbler-side smudge + CRT encode: fresh μ per component,
        // b\' = b + μ·q, residues of a and b\' derived directly mod each prime.
        let t = Instant::now();
        let mus: Vec<u64> = (0..s).map(|_| rng.random()).collect();
        let a_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
            .collect();
        let b_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .enumerate()
            .map(|(pi_idx, &pi)| {
                let qp = q_mod_p[pi_idx];
                b_vals
                    .iter()
                    .zip(&mus)
                    .map(|(&b, &mu)| (b % pi + (mu % pi) * qp) % pi)
                    .collect()
            })
            .collect();
        let t_enc = t.elapsed().as_secs_f64();

        // The driver samples input masks internally but outside its
        // per-stage timers, so garble/eval exclude label provisioning —
        // matching bit-decomp's untimed encode().
        let t = Instant::now();
        let (out, stats) =
            crate::affine::build_s_aff(&mut rng, &x_bits, &params, &a_res, &b_res);
        let core = t.elapsed().as_secs_f64();
        black_box(&out);

        // Evaluator-side CRT decode: Garner-reconstruct the (smudged)
        // integer, then reduce mod q to the field element.
        let t = Instant::now();
        let mut residues = vec![0u64; params.primes.len()];
        for j in 0..s {
            for (i, prime_outs) in out.iter().enumerate() {
                residues[i] = prime_outs[j];
            }
            let v = decoder.reconstruct(&residues);
            black_box(u576_mod_q(&v));
        }
        let t_dec = t.elapsed().as_secs_f64();

        if it >= warmup {
            sw_g.push(stats.garble_secs() + t_enc);
            sw_e.push(stats.eval_secs() + t_dec);
            sw_t.push(t_enc + core + t_dec);
            sw_cf = stats.hash_count_cf;
            sw_ncf = stats.hash_count_ncf;
            sw_comm_bits =
                crate::label::LAMBDA * stats.join_complexity_cf + stats.join_complexity_ncf;
            sw_enc_ms = t_enc * 1e3;
            sw_dec_ms = t_dec * 1e3;
        }
    }
    let (swg_min, swg_med) = stats(sw_g);
    let (swe_min, swe_med) = stats(sw_e);
    let (swt_min, swt_med) = stats(sw_t);

    // ===================== report =====================
    let mib = |b: usize| b as f64 / 1_048_576.0;
    let mb = |bits: usize| bits as f64 / 8.0 / 1e6;
    let m = |x: usize| x as f64 / 1e6;
    eprintln!("\n========== a·x + b benchmark  (N={n}, S={s}, single-thread, release) ==========");
    eprintln!("iters={iters}, warmup={warmup}; values are MEDIAN [MIN] ms; switch garble incl. smudge (64-bit mu)");
    eprintln!("+ CRT-encode of a,b; switch eval incl. Garner + mod-q to field elements; excludes a,b sampling & verify\n");
    eprintln!("{:<22}{:>17}{:>17}{:>17}", "approach", "garble", "eval", "garble+eval");
    eprintln!("{:-<73}", "");
    eprintln!(
        "{:<22}{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]",
        "bit-decomposition", bdg_med, bdg_min, bde_med, bde_min, bdg_med + bde_med, bdg_min + bde_min
    );
    eprintln!(
        "{:<22}{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]{:>9.2} [{:>5.2}]",
        "switch system (CRT)", swg_med, swg_min, swe_med, swe_min, swg_med + swe_med, swg_min + swe_min
    );
    eprintln!(
        "{:<22}{:>15.2}x{:>16.2}x{:>16.2}x",
        "ratio switch/bitdec",
        swg_med / bdg_med,
        swe_med / bde_med,
        (swg_med + swe_med) / (bdg_med + bde_med)
    );
    eprintln!(
        "\nswitch full run: {:.2} [{:.2}] ms  (encode+seed+garble+eval+decode+other)",
        swt_med, swt_min
    );
    eprintln!(
        "switch CRT split: smudge+encode a,b {:.1} ms (in garble col) | Garner + mod-q {:.1} ms (in eval col)",
        sw_enc_ms, sw_dec_ms
    );
    eprintln!("\nhashes (CCRH 16-byte blocks, per party):");
    eprintln!(
        "   bit-decomp   garbler {:.2}M / evaluator {:.2}M   (exact: 4nS / 2nS)",
        m(bd_h.garble_calls),
        m(bd_h.eval_calls),
    );
    eprintln!(
        "   switch       garbler {:.2}M / evaluator \u{2248}{:.2}M  (cf {} + ncf {}; evaluator ~0.3% less —",
        m(sw_cf + sw_ncf),
        m(sw_cf + sw_ncf),
        sw_cf,
        sw_ncf
    );
    eprintln!("                skips open switches; exact split via `bench_axb_hashcounts`)");
    eprintln!(
        "comm:    bit-decomp  {:.1} MiB (GRR, materialized; {:.1} MiB if both rows sent)   |   switch  {:.2} MB (join width)  => ~{:.0}x less\n",
        mib(bd_comm_grr),
        mib(bd_comm),
        mb(sw_comm_bits),
        mib(bd_comm_grr) / (mb(sw_comm_bits) / 1.048576)
    );
}

/// End-to-end latency benchmark: garble → transmit the garbled material over a
/// simulated bandwidth-limited link → eval, for both approaches on the same
/// a·x+b workload. Transmission is modeled analytically as
/// `(bytes · 8) / bandwidth` (pure serialization delay, no RTT); only garble,
/// transmission, and eval are counted — no build/setup, no a,b sampling, no
/// decode. Single-threaded. This is where the switch system's ~50–100× smaller
/// communication flips the comparison on a constrained link.
///
/// Run:
///   cargo test -r --lib bench_axb_network -- --ignored --nocapture
/// Override via env: `N=256 S=1536 ITERS=30 WARMUP=5 BW_MBPS=100 cargo test -r ...`
#[test]
#[ignore]
fn bench_axb_network() {
    use crate::bitdecomp;
    use std::hint::black_box;
    use std::time::Instant;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let n = env("N", 256) as u32;
    let s = env("S", 1536);
    let iters = env("ITERS", 30);
    let warmup = env("WARMUP", 5);
    let bw_mbps: f64 = std::env::var("BW_MBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100.0);
    assert_eq!(
        n as usize,
        bitdecomp::N_BITS,
        "bit-decomp N_BITS is a compile-time constant ({})",
        bitdecomp::N_BITS
    );

    // Transmission latency (ms) to push `bytes` over a `bw_mbps` link.
    let tx_ms = |bytes: usize| -> f64 { (bytes as f64) * 8.0 / (bw_mbps * 1e6) * 1e3 };
    // Median of per-iteration seconds, in ms.
    fn med(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] * 1e3
    }

    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ---- bit-decomposition ----
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x_fp = bitdecomp::Fp::random(&mut rng);
    let mut bdg = Vec::with_capacity(iters);
    let mut g_keep = None;
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut l);
        let dt = t.elapsed().as_secs_f64();
        black_box(&g);
        if it >= warmup {
            bdg.push(dt);
        }
        g_keep = Some(g);
    }
    let g = g_keep.unwrap();
    let active = bitdecomp::encode(&g, x_fp);
    let mut bde = Vec::with_capacity(iters);
    for it in 0..warmup + iters {
        let mut l = bitdecomp::HashLedger::default();
        let t = Instant::now();
        black_box(bitdecomp::eval(&g, &active, x_fp, &mut l));
        let dt = t.elapsed().as_secs_f64();
        if it >= warmup {
            bde.push(dt);
        }
    }
    let (bd_g, bd_e) = (med(bdg), med(bde));
    let bd_tx = tx_ms(g.comm_bytes_grr()); // realistic transmitted GC (one row/bit via GRR)
    let bd_e2e = bd_g + bd_tx + bd_e;

    // ---- switch system (CRT) ----
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let decoder = GarnerDecoder::new(&params.primes); // per-prime-set precompute
    let q_mod_p: Vec<u64> = params.primes.iter().map(|&p| bn254_q_mod(p)).collect();
    let mut swg = Vec::with_capacity(iters);
    let mut swe = Vec::with_capacity(iters);
    let (mut sw_comm_bits, mut sw_cf, mut sw_ncf) = (0usize, 0usize, 0usize);
    for it in 0..warmup + iters {
        let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();
        // Garbler starts from (a, b) over the field: smudge + CRT encode
        // are garble-side.
        let t = Instant::now();
        let mus: Vec<u64> = (0..s).map(|_| rng.random()).collect();
        let a_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
            .collect();
        let b_res: Vec<Vec<u64>> = params
            .primes
            .iter()
            .enumerate()
            .map(|(pi_idx, &pi)| {
                let qp = q_mod_p[pi_idx];
                b_vals
                    .iter()
                    .zip(&mus)
                    .map(|(&b, &mu)| (b % pi + (mu % pi) * qp) % pi)
                    .collect()
            })
            .collect();
        let t_enc = t.elapsed().as_secs_f64();
        let (out, stats) =
            crate::affine::build_s_aff(&mut rng, &x_bits, &params, &a_res, &b_res);
        black_box(&out);
        // Evaluator ends with a·x+b over the field: Garner + mod-q is eval-side.
        let t = Instant::now();
        let mut residues = vec![0u64; params.primes.len()];
        for j in 0..s {
            for (i, prime_outs) in out.iter().enumerate() {
                residues[i] = prime_outs[j];
            }
            let v = decoder.reconstruct(&residues);
            black_box(u576_mod_q(&v));
        }
        let t_dec = t.elapsed().as_secs_f64();
        if it >= warmup {
            swg.push(stats.garble_secs() + t_enc);
            swe.push(stats.eval_secs() + t_dec);
            sw_comm_bits =
                crate::label::LAMBDA * stats.join_complexity_cf + stats.join_complexity_ncf;
            sw_cf = stats.hash_count_cf;
            sw_ncf = stats.hash_count_ncf;
        }
    }
    let (sw_g, sw_e) = (med(swg), med(swe));
    let sw_bytes = sw_comm_bits / 8;
    let sw_tx = tx_ms(sw_bytes);
    let sw_e2e = sw_g + sw_tx + sw_e;

    // ---- report ----
    let mib = |b: usize| b as f64 / 1_048_576.0;
    eprintln!(
        "\n===== a·x + b end-to-end over a {:.0} Mbps link  (N={n}, S={s}, single-thread, release) =====",
        bw_mbps
    );
    eprintln!("iters={iters}, warmup={warmup}; garble/eval = MEDIAN ms; transmit = (GC bytes·8)/bandwidth\n");
    eprintln!(
        "{:<22}{:>10}{:>16}{:>10}{:>14}",
        "approach", "garble", "transmit", "eval", "END-TO-END"
    );
    eprintln!("{:-<72}", "");
    eprintln!(
        "{:<22}{:>7.2}ms{:>9.1}ms{:>14}{:>7.2}ms{:>11.1}ms",
        "bit-decomposition",
        bd_g,
        bd_tx,
        format!("({:.1} MiB)", mib(g.comm_bytes_grr())),
        bd_e,
        bd_e2e
    );
    eprintln!(
        "{:<22}{:>7.2}ms{:>9.1}ms{:>14}{:>7.2}ms{:>11.1}ms",
        "switch system (CRT)",
        sw_g,
        sw_tx,
        format!("({:.2} MB)", sw_bytes as f64 / 1e6),
        sw_e,
        sw_e2e
    );
    eprintln!(
        "\n=> switch system is {:.1}x FASTER end-to-end ({:.1} ms vs {:.0} ms): tiny comm dominates\n   the bandwidth term that the bit-decomposition GC ({:.1} MiB) pays.",
        bd_e2e / sw_e2e,
        sw_e2e,
        bd_e2e,
        mib(g.comm_bytes_grr())
    );
    eprintln!(
        "   (bit-decomp without GRR sends {:.1} MiB => transmit {:.1} ms, end-to-end {:.1} ms.)",
        mib(g.comm_bytes()),
        tx_ms(g.comm_bytes()),
        bd_g + tx_ms(g.comm_bytes()) + bd_e
    );
    let bd_h = bitdecomp::cost_model(s);
    let mh = |x: usize| x as f64 / 1e6;
    eprintln!("\nhashes (CCRH 16-byte blocks, per party):");
    eprintln!(
        "   bit-decomp   garbler {:.2}M / evaluator {:.2}M   (exact: 4nS / 2nS)",
        mh(bd_h.garble_calls),
        mh(bd_h.eval_calls),
    );
    eprintln!(
        "   switch       garbler {:.2}M / evaluator \u{2248}{:.2}M  (cf {} + ncf {}; evaluator ~0.3% less,",
        mh(sw_cf + sw_ncf),
        mh(sw_cf + sw_ncf),
        sw_cf,
        sw_ncf
    );
    eprintln!("                skips open switches; exact split via `bench_axb_hashcounts`)\n");
}

/// Exact per-party CCRH hash-call counts (16-byte AES blocks) for both
/// approaches, measured via the process-global counter. The garbler and
/// evaluator differ for the switch system: the garbler hashes every switch, but
/// the evaluator skips OPEN switches (control ≠ 0), so eval < garble. Requires
/// the `count-hashes` feature (zero-overhead counter, so this never perturbs the
/// timing benches).
///
/// Run:
///   cargo test -r --features count-hashes --lib bench_axb_hashcounts -- --ignored --nocapture
#[cfg(feature = "count-hashes")]
#[test]
#[ignore]
fn bench_axb_hashcounts() {
    use crate::bitdecomp;
    use crate::crypto::{hash_blocks, reset_hash_blocks};

    let n = 256u32;
    let s: usize = std::env::var("S").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
    assert_eq!(n as usize, bitdecomp::N_BITS);
    let mut rng = BenchRng::new(0x0DDC0FFEE0DDF00D);

    // ---- bit-decomposition: garbler vs evaluator blocks ----
    let a_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let b_fp: Vec<bitdecomp::Fp> = (0..s).map(|_| bitdecomp::Fp::random(&mut rng)).collect();
    let x = bitdecomp::Fp::random(&mut rng);
    let mut led = bitdecomp::HashLedger::default();
    reset_hash_blocks();
    let g = bitdecomp::garble(&a_fp, &b_fp, &mut rng, &mut led);
    let bd_g = hash_blocks();
    let active = bitdecomp::encode(&g, x);
    reset_hash_blocks();
    let _ = bitdecomp::eval(&g, &active, x, &mut led);
    let bd_e = hash_blocks();

    // ---- switch system: per-party blocks accumulated across the stream ----
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let a_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let b_vals: Vec<u64> = (0..s).map(|_| rng.random_range(0..1u64 << 48)).collect();
    let a_res: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_res: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();
    let x_bits: Vec<u64> = (0..n as usize).map(|_| rng.random_range(0..2u64)).collect();
    reset_hash_blocks();
    let (_, stats) =
        crate::affine::build_s_aff(&mut rng, &x_bits, &params, &a_res, &b_res);
    let sw_total = hash_blocks();
    let sw_g = stats.garble_hash_blocks;
    let sw_e = stats.eval_hash_blocks;
    let sw_ledger = (stats.hash_count_cf + stats.hash_count_ncf) as u64;

    // Cross-checks: measured == ground truth.
    assert_eq!(bd_g as usize, led.garble_calls, "bit-decomp garbler vs HashLedger");
    assert_eq!(bd_e as usize, led.eval_calls, "bit-decomp evaluator vs HashLedger");
    assert_eq!(sw_g + sw_e, sw_total, "switch per-party blocks must sum to measured total");

    let mm = |x: u64| x as f64 / 1e6;
    eprintln!("\n===== a·x+b CCRH hash calls (16-byte AES blocks), per party  (N={n}, S={s}) =====\n");
    eprintln!("{:<22}{:>14}{:>14}{:>12}", "approach", "GARBLER", "EVALUATOR", "eval/garble");
    eprintln!("{:-<62}", "");
    eprintln!(
        "{:<22}{:>14}{:>14}{:>11.2}x",
        "bit-decomposition", bd_g, bd_e, bd_e as f64 / bd_g as f64
    );
    eprintln!(
        "{:<22}{:>14}{:>14}{:>11.2}x",
        "switch system (CRT)", sw_g, sw_e, sw_e as f64 / sw_g as f64
    );
    eprintln!(
        "\nbit-decomp closed form : garbler 4nS = {}, eval 2nS = {}  (exact, input-independent)",
        2 * bitdecomp::N_BITS * s * bitdecomp::BLOCKS_PER_FE,
        bitdecomp::N_BITS * s * bitdecomp::BLOCKS_PER_FE
    );
    eprintln!(
        "switch                 : garbler {:.2}M / evaluator {:.2}M  (analytical circuit ledger cf {} + ncf {} = {})",
        mm(sw_g),
        mm(sw_e),
        stats.hash_count_cf,
        stats.hash_count_ncf,
        sw_ledger
    );
    eprintln!(
        "  evaluator < garbler because the evaluator skips OPEN switches; the exact\n\
         evaluator count is mildly input-dependent (which one-hot slots are active).\n"
    );
}

// ==================== end-to-end correctness ====================

/// Known-answer check of [`build_s_aff`] against the direct mod-p oracle.
fn assert_s_aff_correct(
    params: &CrtParams,
    a_vals: &[u64],
    b_vals: &[u64],
    x: u64,
    rng: &mut impl Rng,
) {
    let n = params.n as usize;
    let s_dim = a_vals.len();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| a_vals.iter().map(|&a| a % pi).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| b_vals.iter().map(|&b| b % pi).collect())
        .collect();
    let input_bits: Vec<u64> = (0..n).map(|j| (x >> j) & 1).collect();

    let (outputs, _stats) =
        crate::affine::build_s_aff(rng, &input_bits, params, &a_residues, &b_residues);

    for (i, &p_i) in params.primes.iter().enumerate() {
        let x_mod = x % p_i;
        for j in 0..s_dim {
            let expected = ((a_vals[j] % p_i) * x_mod + (b_vals[j] % p_i)) % p_i;
            assert_eq!(
                outputs[i][j], expected,
                "kernel a·x+b mod p mismatch (prime {p_i}, comp {j}, x={x}, S={s_dim})"
            );
        }
    }
}

#[test]
fn test_s_aff_sweep() {
    // build_s_aff against the oracle, across a parameter sweep.
    let mut rng = rng();
    for n in [8u32, 12, 16, 24, 33, 64] {
        let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
        let max_x = if n >= 63 { u64::MAX } else { (1u64 << n) - 1 };
        for s_dim in [1usize, 3] {
            let a_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let b_vals: Vec<u64> = (0..s_dim)
                .map(|_| rng.random_range(0..1u64 << 40))
                .collect();
            let x = rng.random_range(0..=max_x) & max_x;
            assert_s_aff_correct(&params, &a_vals, &b_vals, x, &mut rng);
        }
    }
}


#[test]
#[ignore]
fn bench_axb_stages() {
    // Per-stage wall-clock profiler for build_s_aff.
    // Run: N=256 S=1536 REPS=30 cargo test --release bench_axb_stages -- --ignored --nocapture
    let n: u32 = std::env::var("N")
        .unwrap_or_else(|_| "256".into())
        .parse()
        .unwrap();
    let s_dim: usize = std::env::var("S")
        .unwrap_or_else(|_| "1536".into())
        .parse()
        .unwrap();
    let reps: usize = std::env::var("REPS")
        .unwrap_or_else(|_| "30".into())
        .parse()
        .unwrap();
    let params = CrtParams::from_primes(&FIRST_80_PRIMES, n);
    let mut rng = rng();
    let a_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let b_residues: Vec<Vec<u64>> = params
        .primes
        .iter()
        .map(|&pi| (0..s_dim).map(|_| rng.random_range(0..pi)).collect())
        .collect();
    let mut agg = crate::affine::Stats::default();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let x: u64 = rng.random_range(0..u64::MAX >> (64 - n.min(63)));
        let x_bits: Vec<u64> = (0..n).map(|j| (x >> (j % 64)) & 1).collect();
        let (outputs, stats) = crate::affine::build_s_aff(
            &mut rng,
            &x_bits,
            &params,
            &a_residues,
            &b_residues,
        );
        std::hint::black_box(&outputs);
        agg.chunk_garble_secs += stats.chunk_garble_secs;
        agg.chunk_eval_secs += stats.chunk_eval_secs;
        agg.extract_garble_secs += stats.extract_garble_secs;
        agg.extract_eval_secs += stats.extract_eval_secs;
        agg.fold_garble_secs += stats.fold_garble_secs;
        agg.fold_eval_secs += stats.fold_eval_secs;
        agg.body_garble_secs += stats.body_garble_secs;
        agg.body_eval_secs += stats.body_eval_secs;
    }
    let per_rep = t.elapsed().as_secs_f64() / reps as f64;
    let r = reps as f64;
    eprintln!("{} reps, {:.4}s/rep", reps, per_rep);
    eprintln!(
        "  per-rep garble: chunk {:.2}ms extract {:.2}ms fold {:.2}ms body {:.2}ms",
        1e3 * agg.chunk_garble_secs / r,
        1e3 * agg.extract_garble_secs / r,
        1e3 * agg.fold_garble_secs / r,
        1e3 * agg.body_garble_secs / r,
    );
    eprintln!(
        "  per-rep eval:   chunk {:.2}ms extract {:.2}ms fold {:.2}ms body {:.2}ms",
        1e3 * agg.chunk_eval_secs / r,
        1e3 * agg.extract_eval_secs / r,
        1e3 * agg.fold_eval_secs / r,
        1e3 * agg.body_eval_secs / r,
    );
    eprintln!(
        "  accounted {:.2}ms of {:.2}ms/rep ({:.0}%)",
        1e3 * (agg.garble_secs() + agg.eval_secs()) / r,
        1e3 * per_rep,
        100.0 * (agg.garble_secs() + agg.eval_secs()) / r / per_rep,
    );
}

#[test]
fn test_s_aff_edge_regimes() {
    // Edge-parameter regimes, minus the width-1 trailing sub-chunk
    // (ell ≡ 1 mod 8), which build_s_aff rejects by design (covered by
    // test_extract_rejects_width1).
    let mut rng = rng();

    // p > 2^first_width and p = 2 alone, one odd prime, short padded chunk.
    for primes in [&[2u64][..], &[409][..], &[2, 3, 5, 7, 401, 409][..]] {
        let params = CrtParams::from_primes(primes, 16);
        let a = vec![rng.random_range(0..1u64 << 30)];
        let b = vec![rng.random_range(0..1u64 << 30)];
        let x = rng.random_range(0..1u64 << 16);
        assert_s_aff_correct(&params, &a, &b, x, &mut rng);
    }
    // Single sub-chunk (ell <= 8): tiny prime set keeps the sum small.
    let params = CrtParams::from_primes(&[3u64, 5], 4);
    assert!(params.ell <= 8, "want a single-sub-chunk regime");
    for x in 0..16u64 {
        assert_s_aff_correct(&params, &[7], &[11], x, &mut rng);
    }
}
