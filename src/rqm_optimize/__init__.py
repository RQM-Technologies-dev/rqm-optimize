"""rqm-optimize — optional backend-adjacent SU(2)-aware compression for quantum workflows.

rqm-optimize is a later-stage compression pass that operates on Qiskit
``QuantumCircuit`` objects, downstream of ``rqm-circuits`` (public IR),
``rqm-compiler`` (internal optimization), and usually ``rqm-qiskit`` (lowering).
It does not own the canonical public circuit schema and is not a replacement
for rqm-compiler.

Public API::

    from rqm_optimize import optimize, OptimizationResult

    result = optimize(qc, return_metadata=True)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("rqm-optimize")
except PackageNotFoundError:
    __version__ = "unknown"

from .optimizer import OptimizationResult, optimize

__all__ = ["optimize", "OptimizationResult"]
