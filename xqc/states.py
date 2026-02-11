import jax
import jax.numpy as jnp
from .baseops import Op, ptr as base_ptr


@jax.tree_util.register_pytree_node_class
class State:
    """
    The class State serves as a representation for quantum states, supporting both
    kets (vectors) and density matrices. It is designed to be compatible with the
    Op class and supports partial traces.

    **args**
    arr: A jax array representing the state. Can be a vector (ket) or square matrix (density matrix).
    subs: The subsystem structure submitted as a jax array with the shape [[n1 x n1], [n2, n2], ..., [nN, nN]].
          If None, the system is considered as a single system.
    is_ket: Boolean indicating if the state is a ket. If None, it is inferred from the array shape.
    """
    def __init__(self, arr: jnp.ndarray, subs=None, is_ket=None):
        self.arr = jnp.array(arr, dtype=jnp.complex64)
        
        # Infer is_ket if not provided
        if is_ket is None:
            if self.arr.ndim == 1 or (self.arr.ndim == 2 and self.arr.shape[1] == 1):
                self.is_ket = True
            elif self.arr.ndim == 2 and self.arr.shape[0] == self.arr.shape[1]:
                self.is_ket = False
            else:
                raise ValueError("Input array must be a vector (ket) or a square matrix (density matrix).")
        else:
            self.is_ket = is_ket

        # Standardize ket shape to (N, 1)
        if self.is_ket and self.arr.ndim == 1:
            self.arr = self.arr.reshape(-1, 1)
            
        # Validate density matrix shape
        if not self.is_ket:
            if self.arr.ndim != 2 or self.arr.shape[0] != self.arr.shape[1]:
                raise ValueError("Density matrix must be a square jax.numpy array")

        self.shape = jnp.array(self.arr.shape)

        if subs is None:
            # Default subsystem structure matching Op convention [[N, N]]
            dim = self.arr.shape[0]
            self.subs = jnp.array([[dim, dim]])
        else:
            self.subs = subs

    def __str__(self):
        type_str = "Ket" if self.is_ket else "DensityMatrix"
        return f"State[\ntype={type_str},\nvalue=\n{self.arr},\nshape={self.shape},\nsubsystems=\n{self.subs}\n]"

    def __repr__(self):
        return self.__str__()

    @jax.jit
    def to_dm(self):
        """
        Convert the state to a density matrix.
        If it is already a density matrix, returns self.
        """
        if not self.is_ket:
            return self
        
        # |psi><psi|
        dm_arr = self.arr @ self.arr.conj().T
        return State(dm_arr, subs=self.subs, is_ket=False)

    def ptr(self, keep: jnp.ndarray):
        """
        Calculate the partial trace of the state given an array of the subsystems to retain.
        Always returns a density matrix.

        Args:
          keep: A 1-D jax array that indexes the subsystems to retain

        Returns:
          The reduced state as a State object (density matrix).
        """
        # Convert to density matrix first
        dm_state = self.to_dm()
        
        # Wrap in Op to use baseops.ptr
        op_wrapper = Op(dm_state.arr, dm_state.subs)
        
        # Perform partial trace using baseops implementation
        reduced_op = base_ptr(op_wrapper, keep)
        
        # Return result as a State
        return State(reduced_op.operator, subs=reduced_op.subs, is_ket=False)

    def tree_flatten(self):
        return ((self.arr, self.subs), self.is_ket)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], subs=children[1], is_ket=aux_data)

    def __add__(self, other):
        if isinstance(other, State):
            if self.is_ket != other.is_ket:
                raise ValueError("Cannot add Ket and Density Matrix directly.")
            if not jnp.array_equal(self.subs, other.subs):
                raise ValueError("Subsystem structures must match for addition.")
            return State(self.arr + other.arr, subs=self.subs, is_ket=self.is_ket)
        return NotImplemented

    @jax.jit
    def __mul__(self, other):
        other_arr = jnp.array(other, dtype=jnp.complex64) if not isinstance(other, jnp.ndarray) else other
        return State(self.arr * other_arr, subs=self.subs, is_ket=self.is_ket)

    @jax.jit
    def __rmul__(self, other):
        other_arr = jnp.array(other, dtype=jnp.complex64) if not isinstance(other, jnp.ndarray) else other
        return State(self.arr * other_arr, subs=self.subs, is_ket=self.is_ket)

    def __sub__(self, other):
        if isinstance(other, State):
            if self.is_ket != other.is_ket:
                raise ValueError("Cannot subtract Ket and Density Matrix directly.")
            if not jnp.array_equal(self.subs, other.subs):
                raise ValueError("Subsystem structures must match for subtraction.")
            return State(self.arr - other.arr, subs=self.subs, is_ket=self.is_ket)
        return NotImplemented

    def __matmul__(self, other):
        from .hamiltonian import Hamiltonian
        if isinstance(other, (Op, Hamiltonian)):
            if self.is_ket:
                return NotImplemented
            # DM @ Op -> DM (technically just a matrix, but stored as State(is_ket=False))
            return State(self.arr @ other.operator, subs=self.subs, is_ket=False)
        return NotImplemented

    def __rmatmul__(self, other):
        from .hamiltonian import Hamiltonian
        if isinstance(other, (Op, Hamiltonian)):
            return State(other.operator @ self.arr, subs=self.subs, is_ket=self.is_ket)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, State):
            if self.is_ket != other.is_ket:
                raise ValueError("Cannot subtract Ket and Density Matrix directly.")
            if not jnp.array_equal(self.subs, other.subs):
                raise ValueError("Subsystem structures must match for subtraction.")
            return State(self.arr - other.arr, subs=self.subs, is_ket=self.is_ket)
        return NotImplemented

    def __matmul__(self, other):
        from .hamiltonian import Hamiltonian
        if isinstance(other, (Op, Hamiltonian)):
            if self.is_ket:
                return NotImplemented
            # DM @ Op -> DM (technically just a matrix, but stored as State(is_ket=False))
            return State(self.arr @ other.operator, subs=self.subs, is_ket=False)
        return NotImplemented

    def __rmatmul__(self, other):
        from .hamiltonian import Hamiltonian
        if isinstance(other, (Op, Hamiltonian)):
            return State(other.operator @ self.arr, subs=self.subs, is_ket=self.is_ket)
        return NotImplemented