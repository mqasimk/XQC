# import jax
# import jax.numpy as jnp


# @jax.tree_util.register_pytree_node_class
# class Hamiltonian:
#     def __init__(self, coefs: jnp.ndarray, ops: jnp.ndarray):
#         self.ht = jnp.tensordot(coefs, ops, [[0], [0]])
#     def
#     def tree_flatten(self):
#         return (self.ht,), None
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)


