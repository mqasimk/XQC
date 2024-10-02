import jax
import jax.numpy as jnp
from typeguard import typechecked


# We need to register the class as a valid JAX type through the following wrapper. This tells JAX how to flatten and
# unflatten the object for XLA compilation.
@jax.tree_util.register_pytree_node_class
class Op:
    @typechecked
    def __init__(self, arr: jnp.ndarray, subs: jnp.ndarray):
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("The input array must be a square jax.numpy array")
        self.operator = jnp.array(arr, dtype=jnp.complex64)
        self.shape = jnp.array(arr.shape)
        self.subs = subs
    def __str__(self):
        return f"Op[\nvalue=\n{self.operator},\nshape={self.shape},\nsubsystems=\n{self.subs}\n]"
    def __repr__(self):
        return self.__str__()
    @jax.jit
    def __matmul__(self, other):
        if isinstance(other, Op):
            return Op(self.operator @ other.operator, self.subs)
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
    def tree_flatten(self):
        return (self.operator, self.subs), (self.shape,)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def qOp(arr: jnp.ndarray) -> Op:
    return Op(arr, jnp.array([jnp.array(arr.shape)]))


@jax.jit
def tensor(op1: Op, op2: Op) -> Op:
    #implementation of the same function as qutip has
    if isinstance(op1, Op) and isinstance(op2, Op):
        return Op(jnp.kron(op1.operator, op2.operator), jnp.concatenate([op1.subs, op2.subs], axis=0))
    raise TypeError


@jax.jit
def comm(op1: Op, op2: Op) -> Op:
    return op1@op2-op2@op1


@jax.jit
def acomm(op1: Op, op2: Op) -> Op:
    return op1@op2+op2@op1


@jax.jit
def tr(op: Op):
    return op.tr()



def ptr(op: Op, traceover: jnp.ndarray) -> Op:
    """
    Calculates the partial trace of a density matrix.
    Args:
      rho: A NumPy array representing the density matrix.
      dims: A list of integers representing the dimensions of each subsystem.
      axes: A list of integers representing the indices of the subsystems
            to be traced out.

    Returns:
      A NumPy array representing the reduced density matrix.
    """
    dims = op.subs[:, 0]
    operator = op.operator
    reshaped_operator = jnp.reshape(operator, jnp.concatenate((dims, dims)))
    # Move the subsystems to be traced towards the end
    for axis in sorted(traceover, reverse=True):
        reshaped_operator = jnp.moveaxis(reshaped_operator, axis, -1)
        reshaped_operator = jnp.moveaxis(reshaped_operator, dims.size + axis - 1, -1)
    # Trace over the specified axes
    for _ in traceover:
        reshaped_operator = jnp.trace(reshaped_operator, axis1=-2, axis2=-1)
    return reshaped_operator


@jax.jit
def sx() -> Op:
    return Op(jnp.array([[0, 1],[1, 0]]))


@jax.jit
def sy() -> Op:
    return Op(jnp.array([[0, -1j],[1j, 0]]))


@jax.jit
def sz() -> Op:
    return Op(jnp.array([[1, 0],[0, -1]]))


op1 = qOp(jnp.array([[1, 0],[0, -1]]))
op2 = qOp(jnp.array([[0, 1],[1, 0]]))
print(ptr(tensor(op1, op2), jnp.array([0])))
print(ptr(tensor(op1, op2), jnp.array([1])))

