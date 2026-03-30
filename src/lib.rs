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
mod tests;
