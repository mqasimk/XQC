# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - Unreleased

### Added
- Core `Op` class for quantum operators with JAX-accelerated arithmetic.
- `Hamiltonian` class for building linear combinations of operators.
- `State` class for kets and density matrices with partial trace support.
- `Solver` class with exact diagonalization for time-independent Hamiltonians.
- Pauli operator constructors (`sx`, `sy`, `sz`, `id2`) and `su(n)` basis generation.
- Tensor products (`tensor`), commutators (`comm`), anti-commutators (`acomm`), and partial trace (`ptr`).
- Sphinx documentation with autodoc support.

### Fixed
- Duplicate method definitions in `State` class.
- Incorrect `raise Warning()` in `SwitchingFunction` (now uses `warnings.warn()`).
- Logic error in `SwitchingFunction.__init__` where conditional assignments were overwritten.
- Missing `tree_flatten`/`tree_unflatten` in `SwitchingFunction`.
