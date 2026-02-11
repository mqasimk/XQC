import jax
import jax.numpy as jnp
from .baseops import Op

@jax.tree_util.register_pytree_node_class
class Hamiltonian:
    """
    Represents a Hamiltonian as a linear combination of Op objects.
    Parameters
    ----------
    coefs : array-like
        Coefficients for each term in the Hamiltonian.
    ops : sequence of Op
        Operators corresponding to each coefficient.
    The Hamiltonian is stored internally as an Op (`self.ht`) that
    represents the sum `∑_i coef_i * op_i`.  All arithmetic operations
    delegate to the underlying Op, preserving JAX compatibility.
    """

    def __init__(self, coefs, ops):
        # Convert coefficients to a JAX array of complex64 for consistency
        coefs_arr = jnp.array(coefs, dtype=jnp.complex64)
        if coefs_arr.shape[0] != len(ops):
            raise ValueError(
                "Number of coefficients must match number of operators"
            )
        # Build the linear combination term by term
        self.ht = None
        for c, op in zip(coefs_arr, ops):
            if not isinstance(op, Op):
                raise TypeError("All elements of `ops` must be Op instances")
            term = op * c
            self.ht = term if self.ht is None else self.ht + term

    @property
    def operator(self):
        """Underlying JAX array of the Hamiltonian."""
        return self.ht.operator

    @property
    def subs(self):
        """Subsystem decomposition of the Hamiltonian."""
        return self.ht.subs

    def __repr__(self):
        return f"Hamiltonian({self.ht})"

    def __str__(self):
        return f"Hamiltonian[\n{self.ht}\n]"

    # -----------------------------------------------------------------
    # JAX pytree support
    # -----------------------------------------------------------------
    def tree_flatten(self):
        # The Hamiltonian is flattened to its internal Op.
        return (self.ht,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # `children` contains a single Op instance.
        obj = cls.__new__(cls)
        obj.ht = children[0]
        return obj

    # -----------------------------------------------------------------
    # Arithmetic helpers – delegate to the underlying Op
    # -----------------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Hamiltonian):
            return Hamiltonian.from_op(self.ht + other.ht)
        if isinstance(other, Op):
            return Hamiltonian.from_op(self.ht + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Hamiltonian):
            return Hamiltonian.from_op(self.ht - other.ht)
        if isinstance(other, Op):
            return Hamiltonian.from_op(self.ht - other)
        return NotImplemented

    def __mul__(self, other):
        # Scalar multiplication – JAX will broadcast as needed.
        if isinstance(other, (int, float, complex, jnp.ndarray)):
            return Hamiltonian.from_op(self.ht * other)
        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other):
        # Matrix multiplication with another Op or Hamiltonian.
        if isinstance(other, Hamiltonian):
            return Hamiltonian.from_op(self.ht @ other.ht)
        if isinstance(other, Op):
            return Hamiltonian.from_op(self.ht @ other)
        return NotImplemented

    # -----------------------------------------------------------------
    # Convenience wrappers around the underlying Op methods
    # -----------------------------------------------------------------
    def eigs(self):
        return self.ht.eigs()

    def tr(self):
        return self.ht.tr()

    def T(self):
        return Hamiltonian.from_op(self.ht.T())

    def conj(self):
        return Hamiltonian.from_op(self.ht.conj())

    def dag(self):
        return Hamiltonian.from_op(self.ht.dag())

    @classmethod
    def from_op(cls, op):
        """Create a Hamiltonian directly from an existing Op."""
        obj = cls.__new__(cls)
        obj.ht = op
        return obj
