"""Solvers module for the XQC library.

This module provides placeholder classes for future solver implementations.
"""

import numpy as np


class Solver:
    """Base class for solvers.

    Subclasses should implement the :meth:`solve` method to solve a quantum
    problem using the provided operators and coefficients.
    """

    def __init__(self):
        """Initialize the solver.

        Currently, this base class does not require any initialization
        parameters. Subclasses may extend this constructor.
        """
        pass

    def solve(self, *args, **kwargs):
        """Solve a quantum problem.

        This method should be overridden by subclasses to provide a concrete
        solving algorithm.
        """
        raise NotImplementedError("Solver.solve() must be overridden by a subclass.")
