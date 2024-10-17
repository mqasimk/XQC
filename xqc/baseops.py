import jax
import jax.numpy as jnp


# We need to register the class as a valid JAX type through the following wrapper. This tells JAX how to flatten and
# unflatten the object for XLA compilation.
# We do not need to JIT the methods of the class definition. Any JIT-ed method defined elsewhere will be able to
# construct a pytree node for Op and accelerate the desired operation. JIT compiling class methods becomes
# self-referential.


@jax.tree_util.register_pytree_node_class
class Op:
    """
    The class Op will serve as the base class of quantum objects for this library. It stores the information about
    the array, the shape of the array, and the subsystem decomposition of the array if there is a tensor product
    structure to the problem.

    **args**
    arr: A jax array that is square in shape
    subs: The subsystem structure submitted as a jax array with the shape [[n1 x n1], [n2, n2], ..., [nN, nN]]. If None,
    the system is considered as a single system with structure [n x n].
    """
    def __init__(self, arr: jnp.ndarray, subs = None):
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("The input array must be a square jax.numpy array")
        self.operator = jnp.array(arr, dtype=jnp.complex64)
        self.shape = jnp.array(arr.shape) # shape of the array that makes up the operator
        if subs is None:
            self.subs = jnp.array([jnp.array(arr.shape)])
        else:
            self.subs = subs
    def __str__(self):
        return f"Op[\nvalue=\n{self.operator},\nshape={self.shape},\nsubsystems=\n{self.subs}\n]"
    def __repr__(self):
        return self.__str__()
    @jax.jit
    def __matmul__(self, other):
        if isinstance(other, Op):
            return Op(jnp.matmul(self.operator, other.operator), self.subs)
        return NotImplemented
    @jax.jit
    def __rmatmul__(self, other):
        if isinstance(other, Op):
            return Op(jnp.matmul(other.operator, self.operator), self.subs)
        return NotImplemented
    @jax.jit
    def __mul__(self, other):
        other = jnp.complex64(other)
        if other.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64]:
            return Op(self.operator * other, self.subs)
        return NotImplemented
    @jax.jit
    def __rmul__(self, other):
        other = jnp.complex64(other)
        if other.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64]:
            return Op(self.operator * other, self.subs)
        return NotImplemented
    @jax.jit
    def __add__(self, other):
        if isinstance(other, Op):
            return Op(self.operator + other.operator, self.subs)
        return NotImplemented
    @jax.jit
    def __sub__(self, other):
        if isinstance(other, Op):
            return Op(self.operator - other.operator, self.subs)
        return NotImplemented
    @jax.jit
    def eigs(self):
        jnp.linalg.eigh(self.operator)
    @jax.jit
    def tr(self):
        return self.operator.trace()
    @jax.jit
    def T(self):
        return Op(self.operator.transpose(), self.subs)
    @jax.jit
    def conj(self):
        return Op(self.operator.conj(), self.subs)
    @jax.jit
    def dag(self):
        return Op(self.operator.conj().transpose())
    """
    The flatten and unflatten functions are called by the pytree constructed when JIT is called on an Op object. This
    enables us to JIT any function of the Op class.
    """
    def tree_flatten(self):
        return (self.operator, self.subs), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.jit
def tensor(op1: Op, op2: Op) -> Op:
    """
    Takes in two Op objects with a subsystem structure and takes a kronecker product in the order of input. The
    function returns the larger array with the updated subsystem structure.
    :param op1: First Op object with a subsystem structure sub1.
    :param op2: Second Op object with a subsystem structure sub2.
    :return: New Op object with the updated subsystem structure [sub1[:], sub2[:]]
    """
    #implementation of the same function as qutip
    return Op(jnp.kron(op1.operator, op2.operator), jnp.concatenate([op1.subs, op2.subs], axis=0))


@jax.jit
def comm(op1: Op, op2: Op) -> Op:
    """
    Commutator of two Op objects
    :param op1: First Op object.
    :param op2: Second Op object.
    :return: New Op object constructed out of the commutator of the two sent in.
    """
    return op1@op2-op2@op1


@jax.jit
def acomm(op1: Op, op2: Op) -> Op:
    """
    Anti-commutator of two Op objects
    :param op1: First Op object.
    :param op2: Second Op object.
    :return: New Op object constructed out of the anti-commutator of the two sent in.
    """
    return op1@op2+op2@op1


@jax.jit
def tr(op: Op):
    """
    :param op: Input Op object
    :return: Trace of the operator of the Op object returned as a complex number.
    """
    return op.tr()


def ptr(op: Op, keep: jnp.ndarray) -> Op:
    """
    Calculate the partial trace of a density matrix given an array of the subsystems to retain.

    Args:
      op: Op object with a subsystem structure
      keep: A 1-D jax array that indexes the subsystems to retain

    Returns:
      The reduced operator as an Op object and the updated subsystem structure.
    """
    if op.subs.shape[0] == 1:
        raise AttributeError("There is no subsystem structure to input Op object.")
    if (op.subs.shape[0]-1 < keep).any():
        raise ValueError("The subsystems to keep are out of bounds for the subsystem structure for this system")
    arr = op.operator
    subs_inds = jnp.array(range(op.subs.shape[0]))
    rem = jnp.delete(subs_inds, keep)
    dims = op.subs[:, 0]
    its = rem.shape[0]
    for i in range(its):
        arr = _ptr_util(arr, dims, rem[i])
        dims = jnp.delete(dims, rem[i], axis = 0)
        rem -= 1
    return Op(arr, jnp.delete(op.subs, rem+its, axis=0))


def _ptr_util(arr, dims, rem):
    """
    https://github.com/cvxpy/cvxpy/issues/563 source for this utility function, contributed by user dbunandar

    Takes partial trace over the subsystem defined by 'rem'
    arr: a matrix
    dims: a 1D jax array containing the dimensions of each subsystem
    rem: the indices of the subsytems to be traced out
    """
    dims_ = jnp.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_arr = arr.reshape(jnp.concatenate((dims_, dims_), axis=None))
    # Move the subsystems to be traced towards the end
    reshaped_arr = jnp.moveaxis(reshaped_arr, rem, -1)
    reshaped_arr = jnp.moveaxis(reshaped_arr, len(dims)+rem-1, -1)
    # Trace over the very last row and column indices
    traced_out_arr = jnp.trace(reshaped_arr, axis1=-2, axis2=-1)
    # traced_out_arr is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = jnp.delete(dims_, rem)
    arr_dim = jnp.prod(dims_untraced)
    return traced_out_arr.reshape([arr_dim, arr_dim])


@jax.jit
def sx() -> Op:
    """
    This function constructs a 2x2 pauli-x matrix as an Op object in the computational (z) basis.
    :return: 2x2 pauli-x matrix in the computational (z) basis.
    """
    return Op(jnp.array([[0, 1],[1, 0]]))


@jax.jit
def sy() -> Op:
    """
    This function constructs a 2x2 pauli-y matrix as an Op object in the computational (z) basis.
    :return: 2x2 pauli-y matrix in the computational (z) basis.
    """
    return Op(jnp.array([[0, -1j],[1j, 0]]))


@jax.jit
def sz() -> Op:
    """
    This function constructs a 2x2 pauli-x matrix as an Op object in the computational (z) basis.
    :return: 2x2 pauli-y matrix in the computational (z) basis.
    """
    return Op(jnp.array([[1, 0],[0, -1]]))


@jax.jit
def id2() -> Op:
    return Op(jnp.eye(2))