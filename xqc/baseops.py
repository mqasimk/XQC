import jax
import jax.numpy as jnp
from typeguard import typechecked


# We need to register the class as a valid JAX type through the following wrapper. This tells JAX how to flatten and
# unflatten the object for XLA compilation.
@jax.tree_util.register_pytree_node_class
class Op:
    @typechecked
    def __init__(self, arr: jnp.ndarray):
        if (arr.ndim != 2 or arr.shape[0] != arr.shape[1]):
            raise ValueError("The input array must be a square jax.numpy array")
        self.operator = jnp.array(arr)
        if ((arr.conj()).transpose() == arr).all:
            self.isHerm = True
        else:
            self.isHerm = False
        if self.isHerm:
            if (jnp.linalg.eigh(arr)[0] >= 0).all:
                self.pos = True
            else:
                self.pos = False
        self.shape = arr.shape
    def __str__(self):
        return f"Op[\nvalue=\n{self.operator},\n Hermitian={self.isHerm},\n Positive={self.pos},\nshape={self.shape}\n]"
    def __repr__(self):
        return self.__str__()


    @jax.jit
    def __matmul__(self, other):
        if isinstance(other, Op):
            return Op(self.operator @ other.operator)
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
        if self.isHerm:
            return jnp.linalg.eigh(self.operator)
        else:
            raise ValueError("The Op is not Hermitian and may not have a real spectrum")


    def tree_flatten(self):
        return ((self.operator,), (self.isHerm, self.pos, self.shape))


    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.jit
def tensor(op1, op2):
    #implementation of the same function as qutip has
    if isinstance(op1, Op) and isinstance(op2, Op):
        return Op(jnp.kron(op1.operator, op2.operator))
    else:
        raise TypeError

