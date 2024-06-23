# Vector Calculus Crate
A Rust crate to simplify working with vector calculus, designed to be
as easy to write and work with as possible, to mimic a high-level feel
but low-level Rust features.
## Version Control
- v0.0 - Initial commit: Vectors and Scalar Functions
- v0.01 - Added Vector Functions, as well as curl, div, gradient, etc.
- v0.011 - README includes guide for how to use the crate
- v0.012 - Potential functinos for gradients, and dedicated guide doc
- v0.015 - Parametric curves, sets, and contours added, as well as equality
implemented for vectors
- v0.018 - Added limits for scalar functions
- v0.02 - Evaluate scalar functions on vectors and line integrals for both
scalar functions and vector functions, using Gauss-Legendre, Simpson 1/3 or Riemann.
# Usage
To use the library, the project will have to be using the nightly toolchain,
for which you can copy the "rust-toolchain.toml" file.\
A step by step guide is available in the "guide.md" file.