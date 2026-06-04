/// Affine maps over a primorial ring via CRT decomposition.
pub mod affine;
/// Streaming kernel for the per-prime body batch (System-bypass).
pub mod affine_kernel;
/// Fixed-width 576-bit unsigned integer for CRT output representation.
pub mod bigint;
/// Conversions between binary, word, and ring representations.
pub mod convert;
/// CRT parameters and reconstruction.
pub mod crt;
/// One-hot encoding and scaling.
pub mod ohe;
