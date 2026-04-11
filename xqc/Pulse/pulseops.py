import warnings

import jax


@jax.tree_util.register_pytree_node_class
class SwitchingFunction:
    def __init__(self, tk=None, name=None):
        if tk is None and name is None:
            raise ValueError(
                "You must specify the appropriate pulse vector or the name of the predefined pulse"
            )
        if tk is not None and name is not None:
            warnings.warn(
                "When both the pulse vector and name are specified, the pulse vector is given precedence"
            )
        if tk is not None:
            self.tk = tk
            self.name = "custom sequence"
        else:
            self.tk = tk
            self.name = name

    def tree_flatten(self):
        return (self.tk,), self.name

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.tk = children[0]
        obj.name = aux_data
        return obj
