import jax


@jax.tree_util.register_pytree_node_class
class SwitchingFunction:
    def __init__(self, tk=None, name=None):
        if tk is None and name is None:
            raise ValueError("You must specify the appropriate pulse vector or the name of the predefined pulse")
        if tk is not None and name is not None:
            raise Warning("When both the pulse vector and name are specified, the pulse vector is given precedence")
        if tk is not None:
            self.tk = tk
            self.name = "custom sequence"
        self.tk = tk
        self.name = name
