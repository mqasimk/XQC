import jax
import jax.numpy as jnp
import pytest
from xqc import Hamiltonian, State, sx, sz, Solver
from xqc.solvers import TimeIndependentSolver

def test_solver_exact_ket_sx():
    """Test exact solver with H = X acting on |0>."""
    # H = X
    H = Hamiltonian([1.0], [sx()])
    solver = Solver(H)
    
    # Initial state |0>
    psi0 = State(jnp.array([1, 0]), is_ket=True)
    
    # Time points: 0, pi/2, pi
    ts = jnp.array([0, jnp.pi/2, jnp.pi])
    
    states = solver.solve(psi0, ts, method="exact")
    
    assert len(states) == 3
    
    # t=0: |0>
    assert jnp.allclose(states[0].arr, jnp.array([[1], [0]]))
    
    # t=pi/2: -i|1> (since exp(-i X t) = cos(t)I - i sin(t)X)
    # exp(-i X pi/2) |0> = -i X |0> = -i |1>
    expected_pi_2 = jnp.array([[0], [-1j]], dtype=jnp.complex64)
    assert jnp.allclose(states[1].arr, expected_pi_2, atol=1e-5)
    
    # t=pi: -|0>
    # exp(-i X pi) |0> = -I |0> = -|0>
    expected_pi = jnp.array([[-1], [0]], dtype=jnp.complex64)
    assert jnp.allclose(states[2].arr, expected_pi, atol=1e-5)

def test_solver_exact_dm_sz():
    """Test exact solver with H = Z acting on |+><+|."""
    # H = Z
    H = Hamiltonian([1.0], [sz()])
    solver = Solver(H)
    
    # Initial state |+> = (|0> + |1>)/sqrt(2)
    psi0 = State(jnp.array([1, 1])/jnp.sqrt(2), is_ket=True)
    rho0 = psi0.to_dm()
    
    # Time points: 0, pi/2
    ts = jnp.array([0, jnp.pi/2])
    
    states = solver.solve(rho0, ts, method="exact")
    
    # t=0: |+><+|
    expected_0 = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.complex64)
    assert jnp.allclose(states[0].arr, expected_0, atol=1e-5)
    
    # t=pi/2:
    # U(pi/2) = exp(-i Z pi/2) = -i Z
    # rho(pi/2) = (-i Z) rho (i Z) = Z rho Z
    # Z [[0.5, 0.5], [0.5, 0.5]] Z = [[0.5, -0.5], [-0.5, 0.5]]
    expected_pi_2 = jnp.array([[0.5, -0.5], [-0.5, 0.5]], dtype=jnp.complex64)
    assert jnp.allclose(states[1].arr, expected_pi_2, atol=1e-5)

def test_solver_invalid_method():
    """Test that providing an invalid method raises ValueError."""
    H = Hamiltonian([1.0], [sz()])
    solver = Solver(H)
    psi0 = State(jnp.array([1, 0]), is_ket=True)
    
    with pytest.raises(ValueError, match="Method 'invalid' is not supported"):
        solver.solve(psi0, jnp.array([0.0]), method="invalid")

def test_solver_jit_compatibility():
    """Test that TimeIndependentSolver works within JIT."""
    H = Hamiltonian([1.0], [sx()])
    # Must use TimeIndependentSolver directly as it is the registered Pytree node
    solver = TimeIndependentSolver(H)
    psi0 = State(jnp.array([1, 0]), is_ket=True)
    ts = jnp.array([0.0, jnp.pi/2])
    
    @jax.jit
    def run_evolution(s, state, t):
        # solve returns a list of States
        results = s.solve(state, t)
        # Return the array of the last state to verify
        return results[-1].arr

    final_arr = run_evolution(solver, psi0, ts)
    
    # Expected at pi/2: -i|1>
    expected = jnp.array([[0], [-1j]], dtype=jnp.complex64)
    assert jnp.allclose(final_arr, expected, atol=1e-5)