# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XQC is a quantum simulation library built with JAX to accelerate noisy simulations of open quantum systems and quantum circuits. Python 3.11+ with JAX (CUDA 12.2).

## Commands

- **Run all tests:** `pytest -q`
- **Run a single test file:** `pytest -q xqc/test_solvers.py`
- **Build docs:** `make docs` (or `python docs/generate_docs.py`)
- **Clean docs:** `make clean`
- **Install dev mode:** `pip install -e .`

## Architecture

The library follows a layered composition model where higher layers build on lower ones:

```
baseops.py (Op)  →  hamiltonian.py (Hamiltonian)  →  states.py (State)  →  solvers.py (Solver)
```

- **`xqc/baseops.py`** — `Op` class: base quantum operator wrapping a square JAX array. Provides Pauli constructors (`sx`, `sy`, `sz`, `id2`), `tensor` (Kronecker product), `comm`/`acomm` (commutators), `tr`, `ptr` (partial trace), and `su(n)` (Pauli basis generator).
- **`xqc/hamiltonian.py`** — `Hamiltonian` class: linear combination of `Op` objects (`Hamiltonian(coefs, ops)`). Stores the sum internally as a single `Op` (`self.ht`). Has a `from_op` classmethod for wrapping an existing `Op`.
- **`xqc/states.py`** — `State` class: represents kets (column vectors) and density matrices. Kets are auto-reshaped to `(N, 1)`. Supports `to_dm()` conversion and partial trace via `ptr(keep)`.
- **`xqc/solvers.py`** — `Solver` dispatches to `TimeIndependentSolver`, which pre-computes eigendecomposition and evolves states via `jax.vmap` over time points. `_evolve` is a static JIT-compiled method with `static_argnums=(4,)` for the `is_ket` flag.
- **`xqc/Pulse/pulseops.py`** — `SwitchingFunction` class for pulse control (in development, not yet integrated into the main API).

## Key Conventions

- All core classes (`Op`, `Hamiltonian`, `State`, `TimeIndependentSolver`) are registered as **JAX pytree nodes** using `@jax.tree_util.register_pytree_node_class`. Each implements `tree_flatten`/`tree_unflatten`.
- Performance-critical methods use `@jax.jit`. When JIT-compiling with boolean/integer flags, use `static_argnums`.
- All arrays use `dtype=jnp.complex64`.
- The public API is exported from `xqc/__init__.py`.
- Tests are co-located with source code in `xqc/` (files named `test_*.py`).
- Documentation uses Sphinx with autodoc/napoleon extensions, generated from docstrings.
- CI (GitHub Actions) only builds documentation on push/PR to `main` — no automated test runs in CI.
- No linter or formatter is configured.
