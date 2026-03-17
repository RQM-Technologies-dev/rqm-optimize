"""rqm-optimize — SU(2)-aware circuit compression for quantum workflows.

Public API::

    from rqm_optimize import optimize, OptimizationResult

    result = optimize(qc, return_metadata=True)
"""

from .optimizer import OptimizationResult, optimize

__all__ = ["optimize", "OptimizationResult"]
