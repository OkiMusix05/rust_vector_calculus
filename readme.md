# Vector Calculus Crate

A Rust crate to simplify working with vector calculus, designed to be
as easy to write and work with as possible, to mimic a high-level feel
but low-level Rust features.

# Usage
To use the library, the project will have to be using the nightly toolchain,
for which you can copy the [rust-toolchain.toml](rust-toolchain.toml) file.

A guide with all the features included is also available in the
[guide.md](guide.md) file. Here you can learn about all of the macros provided.
If you don't want to keep coming to github, there's also the
[vector calculus crate](http://periodic-move-478.notion.site) website, which is the same.

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
- v0.021 - Single variable scalar functions implemented, with derivatives and 
the `integral!` macro
- v0.022 - Added parametric surfaces
- v0.023/v0.024 - Advancements on double integrals
- v0.03 - Double integrals with non-constant bounds and triple integrals with 
constant bounds
- v0.035 - Surface integrals for scalar functions skeleton working, no macro yet
- v0.04 - Surface integrals for scalar and vector functions, reworking of vector
functions and a macro. Area method for surfaces
- v0.045 - Reworked Parametric Curves and Contours, as well as added the length method
to contours.
- v0.05 - Generalized the integral macro, addded macros for sin, cos, tan, and ln, 
added multiple double integration methods, and a setup macro for importing the 
IntegrationMethod and MultipleIntegrationMethod's variants, as well as pi and e.
- v0.051 - Some restructuring, generalized the near! macro.
- v0.052 - Fixed some things, created some documentation for docs.rs
- v0.053 - More doc, fixed errors.