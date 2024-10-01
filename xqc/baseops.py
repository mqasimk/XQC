import jax.numpy as jnp
from typeguard import typechecked


def is_square_2d_array(arr):
    # Check if the array has 2 dimensions
    if arr.ndim != 2:
        return False
    # Check if the array is square
    if arr.shape[0] != arr.shape[1]:
        return False
    return True


class Op:
    @typechecked
    def __init__(self, arr: jnp.ndarray):
        if not is_square_2d_array(arr):
            raise ValueError("The input array must be a square jax.numpy array")
        self.operator = jnp.array(arr)
        if jnp.all((arr.conj()).transpose() == arr):
            self.isHerm = True
        else:
            self.isHerm = False
        if self.isHerm:
            if jnp.all(jnp.linalg.eigh(arr)[0] >= 0):
                self.pos = True
            else:
                self.pos = False
        self.shape = arr.shape
    def __str__(self):
        return f"op(\nvalue=\n{self.operator},\n Hermitian={self.isHerm},\n Positive={self.pos},\nshape={self.shape})"
    def __repr__(self):
        return self.__str__()
    def __matmul__(self, other):
        if isinstance(other, Op):
            return Op(self.operator @ other.operator)
        return NotImplemented
    def __add__(self, other):
        if isinstance(other, Op):
            return Op(self.operator + other.operator)
        return NotImplemented
    def __sub__(self, other):
        if isinstance(other, Op):
            return Op(self.operator - other.operator)
        return NotImplemented
    def eigs(self):
        if self.isHerm:
            return jnp.linalg.eigh(self.operator)