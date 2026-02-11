# XQC
A quantum simulation library built ground up with JAX to accelerate noisy simulations of open quantum systems and
quantum circuits.

The current build is built and tested on python 3.11.8 packaged by conda-forge running jax cuda 0.4.24 with jaxlib built
with cuda 12.2.

## Documentation

The library uses Sphinx to generate documentation from the docstrings. To build the HTML documentation, run the following command:

```bash
python docs/generate_docs.py
```

The generated HTML files will be placed in `docs/_build`. Open `docs/_build/index.html` in a browser to view the documentation.

You can also run the script directly via the clickable reference [`generate_docs.py`](docs/generate_docs.py:1).

## Features

- Core quantum operator class [`Op`](xqc/baseops.py:13) with JAX‑accelerated arithmetic, tensor products, commutators, and partial trace.
- [`Hamiltonian`](xqc/hamiltonian.py:5) class for building linear combinations of operators.
- `State` class for representing quantum states (kets and density matrices) with support for partial traces.
- `Solver` class providing exact diagonalization methods for time-independent Hamiltonians.
- Comprehensive test suite covering operators, Hamiltonians, and solvers (`test_baseops_extended.py`, `test_hamiltonian.py`, `test_solvers.py`).
- Sphinx documentation generated from docstrings.

## Installation

```bash
conda install -c conda-forge jax cuda=12.2
pip install -e .
```

## Quick start

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

## Development status

The library is **actively under development**. Core functionality for quantum operators and Hamiltonians is stable and covered by tests. Solver implementations are planned for future releases.

## Testing

Run the test suite with:

```bash
pytest -q
```

## Documentation

Generate the latest documentation with:

```bash
python docs/generate_docs.py
```

## Contributing

Contributions are welcome! Please open issues or pull requests on the repository.