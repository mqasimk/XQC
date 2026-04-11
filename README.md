# XQC

A quantum simulation library built from the ground up with JAX to accelerate noisy simulations of open quantum systems and quantum circuits.

## Features

- Core quantum operator class [`Op`](xqc/baseops.py) with JAX-accelerated arithmetic, tensor products, commutators, and partial trace.
- [`Hamiltonian`](xqc/hamiltonian.py) class for building linear combinations of operators.
- [`State`](xqc/states.py) class for representing quantum states (kets and density matrices) with support for partial traces.
- [`Solver`](xqc/solvers.py) class providing exact diagonalization methods for time-independent Hamiltonians.
- Sphinx documentation generated from docstrings.

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

For GPU support with CUDA:

```bash
conda install -c conda-forge jax cuda=12.2
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from xqc import sx, sz, Op, Hamiltonian, State, Solver

# Define Pauli operators
op_x = sx()
op_z = sz()

# Build a Hamiltonian H = 0.5*X + 1.5*Z
ham = Hamiltonian([0.5, 1.5], [op_x, op_z])
print("Hamiltonian matrix:", ham.operator)

# Define an initial state |0>
psi0 = State(jnp.array([1, 0]), is_ket=True)

# Evolve the state
solver = Solver(ham)
ts = jnp.linspace(0, 1.0, 10)
states = solver.solve(psi0, ts)
print("Final state:", states[-1])
```

## Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest tests/ -v
```

Lint and format:

```bash
ruff check xqc/
ruff format xqc/
```

Or use the Makefile shortcuts:

```bash
make install   # Install with dev dependencies
make test      # Run tests
make lint      # Run linter
make format    # Auto-format code
make docs      # Build Sphinx documentation
```

## Documentation

The library uses Sphinx to generate documentation from docstrings:

```bash
python docs/generate_docs.py
```

The generated HTML files will be placed in `docs/_build`. Open `docs/_build/index.html` in a browser to view the documentation.

## Development Status

The library is **actively under development**. Core functionality for quantum operators and Hamiltonians is stable and covered by tests. Solver implementations are planned for future releases.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
