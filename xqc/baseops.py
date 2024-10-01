import jax
import jax.numpy as jnp
from typeguard import typechecked


# We need to register the class as a valid JAX type through the following wrapper. This tells JAX how to flatten and
# unflatten the object for XLA compilation.
@jax.tree_util.register_pytree_node_class
class Op:
    @typechecked
    def __init__(self, arr: jnp.ndarray):
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("The input array must be a square jax.numpy array")
        self.operator = jnp.array(arr, dtype=jnp.complex64)
        self.shape = arr.shape
    def __str__(self):
        return f"Op[\nvalue=\n{self.operator},\nshape={self.shape}\n]"
    def __repr__(self):
        return self.__str__()
    @jax.jit
    def __matmul__(self, other):
        if isinstance(other, Op):
            return Op(self.operator @ other.operator)
        return NotImplemented
    @jax.jit
    def __mul__(self, other):
        other = jnp.complex64(other)
        if other.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64]:
            return Op(self.operator * other)
        return NotImplemented
    @jax.jit
    def __rmul__(self, other):
        other = jnp.complex64(other)
        if other.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64]:
            return Op(self.operator * other)
        return NotImplemented
    @jax.jit
    def __add__(self, other):
        if isinstance(other, Op):
            return Op(self.operator + other.operator)
        return NotImplemented
    @jax.jit
    def __sub__(self, other):
        if isinstance(other, Op):
            return Op(self.operator - other.operator)
        return NotImplemented
    @jax.jit
    def eigs(self):
        jnp.linalg.eigh(self.operator)
    def tree_flatten(self):
        return (self.operator,), (self.shape,)
    @jax.jit
    def T(self):
        return Op(self.operator.transpose())
    @jax.jit
    def conj(self):
        return Op(self.operator.conj())
    @jax.jit
    def dag(self):
        return Op(self.operator.conj().transpose())
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.jit
def tensor(op1: Op, op2: Op) -> Op:
    #implementation of the same function as qutip has
    if isinstance(op1, Op) and isinstance(op2, Op):
        return Op(jnp.kron(op1.operator, op2.operator))
    else:
        raise TypeError


@jax.jit
def comm(op1: Op, op2: Op) -> Op:
    return op1@op2-op2@op1


@jax.jit
def acomm(op1: Op, op2: Op) -> Op:
    return op1@op2+op2@op1


@jax.jit
def sx() -> Op:
    return Op(jnp.array([[0, 1],[1, 0]]))


@jax.jit
def sy() -> Op:
    return Op(jnp.array([[0, -1j],[1j, 0]]))


@jax.jit
def sz() -> Op:
    return Op(jnp.array([[1, 0],[0, -1]]))

