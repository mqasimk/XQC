# Contributing to XQC

Thanks for your interest in contributing to XQC! Here's how to get started.

## Development Setup

1. Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/mqasimk/XQC.git
cd XQC
pip install -e ".[dev]"
```

2. Verify your setup by running the tests:

```bash
python -m pytest tests/ -v
```

## Making Changes

1. Create a branch from `main` for your work.
2. Make your changes, keeping commits focused and descriptive.
3. Add or update tests in `tests/` for any new functionality.
4. Ensure all checks pass before submitting:

```bash
python -m pytest tests/ -v   # Tests pass
ruff check xqc/              # No lint errors
ruff format --check xqc/     # Code is formatted
```

## Code Style

- Code is linted and formatted with [Ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`.
- Follow existing patterns in the codebase.
- All core classes should be registered as JAX pytrees (`@jax.tree_util.register_pytree_node_class`) with `tree_flatten` and `tree_unflatten` methods.

## Submitting a Pull Request

1. Push your branch and open a pull request against `main`.
2. Describe what your change does and why.
3. Link any related issues.
4. CI will automatically run tests and linting on your PR.

## Reporting Issues

Open a GitHub issue with:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Your environment: Python version, JAX version, OS.
