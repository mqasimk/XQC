"""Solvers module for the XQC library.

This module provides solver classes for quantum dynamics.
"""

import functools
import jax
import jax.numpy as jnp
from .hamiltonian import Hamiltonian
from .states import State

class Solver:
    """General solver class for quantum dynamics.

    This class serves as an interface to various solving algorithms.
    """

    def __init__(self, H: Hamiltonian):
        """Initialize the solver.

        Args:
            H: The Hamiltonian of the system.
        """
        self.H = H

    def solve(self, state0: State, ts: jnp.ndarray, method: str = "exact") -> list[State]:
        """Solve a quantum problem.

        Args:
            state0: The initial quantum state.
            ts: A sequence of time points.
            method: The solving method to use. Default is "exact".

        Returns:
            A list of State objects representing the time evolution.
        """
        if method == "exact":
            return TimeIndependentSolver(self.H).solve(state0, ts)
        else:
            raise ValueError(f"Method '{method}' is not supported.")


@jax.tree_util.register_pytree_node_class
class TimeIndependentSolver(Solver):
    """
    Solver for the time-independent Schrödinger equation.

    This solver computes the time evolution of a quantum state (ket or density matrix)
    under a time-independent Hamiltonian using exact diagonalization.
    """

    def __init__(self, H: Hamiltonian):
        """
        Initialize the solver with a Hamiltonian.

        Args:
            H: The time-independent Hamiltonian of the system.
        """
        super().__init__(H)
        # Pre-compute diagonalization for efficiency
        self.evals, self.evecs = self.H.eigs()

    def tree_flatten(self):
        return ((self.H, self.evals, self.evecs), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.H = children[0]
        obj.evals = children[1]
        obj.evecs = children[2]
        return obj

    def solve(self, state0: State, ts: jnp.ndarray) -> list[State]:
        """
        Evolve the initial state over the specified time points.

        Args:
            state0: The initial quantum state (State object).
            ts: A sequence of time points (array-like).

        Returns:
            A list of State objects corresponding to the state at each time point.
        """
        ts_arr = jnp.array(ts)
        # Ensure ts is at least 1D
        if ts_arr.ndim == 0:
            ts_arr = ts_arr.reshape(1)

        evolved_arrays = self._evolve(self.evals, self.evecs, state0.arr, ts_arr, state0.is_ket)
        
        return [State(arr, subs=state0.subs, is_ket=state0.is_ket) for arr in evolved_arrays]

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(4,))
    def _evolve(evals, evecs, arr0, ts, is_ket):
        v = evecs
        v_dag = v.conj().T

        if is_ket:
            state0_eig = v_dag @ arr0
        else:
            state0_eig = v_dag @ arr0 @ v

        def step(t):
            exp_diag = jnp.exp(-1j * evals * t)
            if is_ket:
                return v @ (state0_eig * exp_diag.reshape(-1, 1))
            else:
                factor = exp_diag[:, None] * exp_diag.conj()[None, :]
                return v @ (state0_eig * factor) @ v_dag
        
        return jax.vmap(step)(ts)
