import pytest
import jax.numpy as jnp
from xqc.baseops import Op, sx, sy, sz, id2, tensor, comm, acomm, tr, ptr, su


def test_op_initialization_and_subs():
    # Square matrix should initialize correctly
    arr = jnp.eye(2)
    op = Op(arr)
    assert jnp.array_equal(op.operator, arr)
    # Default subsystem structure should be a single entry matching the matrix dimensions
    assert op.subs.shape == (1, 2)
    assert op.subs[0][0] == 2 and op.subs[0][1] == 2

    # Non‑square matrix should raise a ValueError
    with pytest.raises(ValueError):
        Op(jnp.array([1, 2, 3]))


def test_op_scalar_multiplication_and_addition():
    op = Op(jnp.eye(2))
    # scalar multiplication
    op2 = op * 3
    assert jnp.allclose(op2.operator, jnp.eye(2) * 3)

    # addition with matching subsystems
    op3 = Op(jnp.array([[0, 1], [1, 0]]))
    op_sum = op + op3
    expected_sum = jnp.eye(2) + jnp.array([[0, 1], [1, 0]])
    assert jnp.allclose(op_sum.operator, expected_sum)

    # addition with mismatched subsystems should raise ValueError
    op_big = Op(jnp.eye(4), subs=jnp.array([[2, 2], [2, 2]]))
    with pytest.raises(ValueError):
        op + op_big

    # subtraction
    op_sub = op - op3
    expected_sub = jnp.eye(2) - jnp.array([[0, 1], [1, 0]])
    assert jnp.allclose(op_sub.operator, expected_sub)

    # matrix multiplication
    op_mat = op @ op3
    expected_mat = jnp.eye(2) @ jnp.array([[0, 1], [1, 0]])
    assert jnp.allclose(op_mat.operator, expected_mat)


def test_op_conjugate_transpose_and_dagger():
    op = Op(jnp.array([[0, 1j], [-1j, 0]]))
    # dagger should be conjugate transpose
    dag = op.dag()
    assert jnp.allclose(dag.operator, op.operator.conj().T)
    # transpose
    trans = op.T()
    assert jnp.allclose(trans.operator, op.operator.T)
    # conjugate
    conj = op.conj()
    assert jnp.allclose(conj.operator, op.operator.conj())


def test_tensor_and_commutator_functions():
    op_a = sx()
    op_b = sz()
    # tensor product
    tens = tensor(op_a, op_b)
    expected_tensor = jnp.kron(op_a.operator, op_b.operator)
    assert jnp.allclose(tens.operator, expected_tensor)
    # subsystems should be concatenated
    assert jnp.array_equal(tens.subs, jnp.concatenate([op_a.subs, op_b.subs], axis=0))

    # commutator
    comm_op = comm(op_a, op_b)
    expected_comm = op_a @ op_b - op_b @ op_a
    assert jnp.allclose(comm_op.operator, expected_comm.operator)

    # anti‑commutator
    acomm_op = acomm(op_a, op_b)
    expected_acomm = op_a @ op_b + op_b @ op_a
    assert jnp.allclose(acomm_op.operator, expected_acomm.operator)


def test_trace_and_partial_trace():
    op = sx()
    # trace of a Pauli‑X is zero
    assert tr(op) == op.tr()

    # partial trace on a 2‑qubit system, keep subsystem 0
    op_2q = Op(jnp.kron(jnp.array([[1, 0], [0, 0]]), jnp.eye(2)), subs=jnp.array([[2, 2], [2, 2]]))
    ptr_op = ptr(op_2q, jnp.array([0]))
    # Result should be a 2x2 matrix (the reduced density matrix of subsystem 0)
    assert ptr_op.operator.shape == (2, 2)


def test_su_basis_generation():
    # n = 1 should produce 4 basis elements of shape (2, 2)
    basis1 = su(1)
    assert basis1.shape == (4, 2, 2)
    # n = 2 should produce 16 basis elements of shape (4, 4)
    basis2 = su(2)
    assert basis2.shape == (16, 4, 4)
