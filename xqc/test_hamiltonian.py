import pytest
import jax.numpy as jnp
from xqc.baseops import Op, sx, sz
from xqc.hamiltonian import Hamiltonian

def test_hamiltonian_initialization_and_properties():
    # Define two Pauli operators
    op1 = sx()
    op2 = sz()
    # Coefficients
    coefs = [0.5, 1.5]
    ham = Hamiltonian(coefs, [op1, op2])
    # Expected operator = 0.5*op1 + 1.5*op2
    expected = op1 * 0.5 + op2 * 1.5
    assert jnp.allclose(ham.operator, expected.operator)
    assert jnp.array_equal(ham.subs, expected.subs)

def test_hamiltonian_invalid_initialization():
    op = sx()
    # Mismatched lengths
    with pytest.raises(ValueError):
        Hamiltonian([1.0], [op, op])
    # Non-Op element
    with pytest.raises(TypeError):
        Hamiltonian([1.0, 2.0], [op, "not an Op"])

def test_hamiltonian_arithmetic():
    op1 = sx()
    op2 = sz()
    ham1 = Hamiltonian([1.0], [op1])
    ham2 = Hamiltonian([2.0], [op2])
    # addition
    ham_sum = ham1 + ham2
    expected_sum = op1 + op2 * 2.0
    assert jnp.allclose(ham_sum.operator, expected_sum.operator)
    # subtraction
    ham_diff = ham2 - ham1
    expected_diff = op2 * 2.0 - op1
    assert jnp.allclose(ham_diff.operator, expected_diff.operator)
    # scalar multiplication
    ham_scaled = 3 * ham1
    expected_scaled = op1 * 3.0
    assert jnp.allclose(ham_scaled.operator, expected_scaled.operator)
    # matrix multiplication
    ham_mat = ham1 @ ham2
    expected_mat = op1 @ (op2 * 2.0)
    assert jnp.allclose(ham_mat.operator, expected_mat.operator)

def test_hamiltonian_methods():
    op = sx()
    ham = Hamiltonian([1.0], [op])
    eigvals, eigvecs = ham.eigs()
    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)
    # trace
    assert ham.tr() == op.tr()
    # transpose
    assert jnp.allclose(ham.T().operator, op.T().operator)
    # conjugate
    assert jnp.allclose(ham.conj().operator, op.conj().operator)
    # dagger
    assert jnp.allclose(ham.dag().operator, op.dag().operator)
