import jax.numpy as jnp
import pytest
from xqc import Op, Hamiltonian, sx, sz, sy
from xqc.states import State

def test_hamiltonian_ket_multiplication():
    """Test Hamiltonian acting on a Ket state."""
    # H = Z
    H = Hamiltonian([1.0], [sz()])
    # |0>
    psi = State(jnp.array([1, 0]), is_ket=True)
    
    # H|0> = |0>
    res = H @ psi
    assert isinstance(res, State)
    assert res.is_ket
    assert jnp.allclose(res.arr, jnp.array([[1], [0]]))
    
    # |1>
    psi_1 = State(jnp.array([0, 1]), is_ket=True)
    # H|1> = -|1>
    res_1 = H @ psi_1
    assert jnp.allclose(res_1.arr, jnp.array([[0], [-1]]))

def test_hamiltonian_dm_multiplication():
    """Test Hamiltonian acting on a Density Matrix."""
    # H = X
    H = Hamiltonian([1.0], [sx()])
    # rho = |0><0|
    psi = State(jnp.array([1, 0]), is_ket=True)
    rho = psi.to_dm()
    
    # H rho = X |0><0| = |1><0|
    # |1><0| = [[0, 0], [1, 0]]
    res = H @ rho
    assert not res.is_ket
    expected = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)
    assert jnp.allclose(res.arr, expected)
    
    # rho H = |0><0| X = |0><1|
    # |0><1| = [[0, 1], [0, 0]]
    res_right = rho @ H
    expected_right = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)
    assert jnp.allclose(res_right.arr, expected_right)

def test_op_state_interaction():
    """Test raw Op acting on State."""
    op = sy()
    psi = State(jnp.array([1, 0]), is_ket=True)
    
    # Y |0> = i|1>
    res = op @ psi
    expected = jnp.array([[0], [1j]], dtype=jnp.complex64)
    assert jnp.allclose(res.arr, expected)

def test_state_subtraction():
    """Test subtraction of States (needed for commutators)."""
    psi = State(jnp.array([1, 0]), is_ket=True)
    rho = psi.to_dm()
    
    # rho - rho = 0
    diff = rho - rho
    assert jnp.allclose(diff.arr, jnp.zeros((2, 2)))

def test_commutator_simulation():
    """Test calculating a commutator [H, rho] using State arithmetic."""
    # H = Z
    H = Hamiltonian([1.0], [sz()])
    # |+> = (|0> + |1>)/sqrt(2)
    psi = State(jnp.array([1, 1])/jnp.sqrt(2), is_ket=True)
    rho = psi.to_dm() 
    # rho = 0.5 * [[1, 1], [1, 1]]
    
    # [Z, rho] = Z rho - rho Z
    # Z rho = 0.5 * [[1, 1], [-1, -1]]
    # rho Z = 0.5 * [[1, -1], [1, -1]]
    # Comm = 0.5 * [[0, 2], [-2, 0]] = [[0, 1], [-1, 0]]
    
    comm = (H @ rho) - (rho @ H)
    
    expected = jnp.array([[0, 1], [-1, 0]], dtype=jnp.complex64)
    assert jnp.allclose(comm.arr, expected)

def test_ptr_interop():
    """Test partial trace functionality via State.ptr."""
    # Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    bell = jnp.array([1, 0, 0, 1]) / jnp.sqrt(2)
    psi = State(bell, subs=jnp.array([[2, 2], [2, 2]]), is_ket=True)
    
    # Trace out qubit 1 (index 1), keep qubit 0 (index 0)
    rho_A = psi.ptr(keep=jnp.array([0]))
    
    # Expect I/2
    assert jnp.allclose(rho_A.arr, 0.5 * jnp.eye(2))
    assert rho_A.subs.shape == (1, 2)
    assert rho_A.subs[0, 0] == 2

def test_hamiltonian_composition_and_state():
    """Test Hamiltonian composed of multiple terms acting on State."""
    # H = 0.5*X + 0.5*Z
    H = Hamiltonian([0.5, 0.5], [sx(), sz()])
    
    # |0>
    psi = State(jnp.array([1, 0]), is_ket=True)
    
    # H|0> = 0.5*|1> + 0.5*|0> = 0.5*[1, 1]^T
    res = H @ psi
    expected = jnp.array([[0.5], [0.5]], dtype=jnp.complex64)
    assert jnp.allclose(res.arr, expected)