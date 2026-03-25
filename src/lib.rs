//! Switch system framework for garbled arithmetic circuits.

/// Higher-level circuit components built on the core system.
pub mod components;
/// Concrete execution (propagation of values) of a system.
pub mod exec;
/// The constraint system: wire allocation, gate construction, and derived operations.
pub mod system;
/// Core types: values, wires, gates, and arithmetic over rings.
pub mod types;

#[cfg(test)]
mod tests {
    use crate::components::convert::{word_to_hot, word_to_ring};
    use crate::exec::Exec;
    use crate::system::System;
    use crate::types::Val;

    #[test]
    fn word_to_hot_4bit() {
        let mut sys = System::new();
        let x = sys.input_bits(4);
        let hot = word_to_hot(&mut sys, x);

        let mut exec = Exec::new(&sys);
        exec.set(x, Val::from_bits(11, 4));
        exec.run();

        for (i, &w) in hot.iter().enumerate() {
            let v = exec.get(w);
            if i == 11 {
                assert_eq!(v, Val::new(1, 16), "hot[{i}] should be 1");
            } else {
                assert_eq!(v, Val::new(0, 16), "hot[{i}] should be 0");
            }
        }
    }

    #[test]
    fn word_to_ring_square() {
        // g(x) = x^2 mod 32, evaluated over a 3-bit input (Z_8 → Z_32)
        let truth_table: Vec<u64> = (0u64..8).map(|x| (x * x) % 32).collect();
        let r_mod = 32u64;

        for input in 0u64..8 {
            let mut sys = System::new();
            let x = sys.input_bits(3);
            let out = word_to_ring(&mut sys, x, &truth_table, r_mod);

            let mut exec = Exec::new(&sys);
            exec.set(x, Val::from_bits(input, 3));
            exec.run();

            let expected = (input * input) % r_mod;
            assert_eq!(
                exec.get(out),
                Val::new(expected, r_mod),
                "g({input}) should be {expected}"
            );
        }
    }
}
