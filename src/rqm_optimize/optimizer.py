"""optimizer.py — Public API, type dispatch, strategy validation, and result
packaging for rqm-optimize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from qiskit import QuantumCircuit


@dataclass
class OptimizationResult:
    """Container for the output of :func:`optimize` when ``return_metadata=True``.

    Attributes:
        circuit: The optimized ``QuantumCircuit``.
        original_gate_count: Gate count of the input circuit.
        optimized_gate_count: Gate count of the output circuit.
        fused_runs: Number of single-qubit runs that were fused (≥ 2 gates
            collapsed into one).
        strategy: Name of the optimization strategy applied.
        notes: Human-readable notes about the optimization, if any.
    """

    circuit: QuantumCircuit
    original_gate_count: int
    optimized_gate_count: int
    fused_runs: int
    strategy: str
    notes: list[str] = field(default_factory=list)


_SUPPORTED_STRATEGIES = frozenset({"geodesic"})


def optimize(
    circuit: Any,
    backend: Any = None,
    strategy: str = "geodesic",
    return_metadata: bool = False,
) -> Union[QuantumCircuit, OptimizationResult]:
    """Optimize a quantum circuit by compressing single-qubit gate runs.

    Scans the circuit for contiguous runs of single-qubit unitary gates on
    each qubit, fuses each run into a single SU(2)-equivalent operation, and
    returns a simplified circuit.

    Non-single-qubit operations (multi-qubit gates, measurements, barriers,
    resets) are left in place and act as fusion boundaries.

    The returned circuit is always a *new* object — the input is never mutated.

    Args:
        circuit: A Qiskit ``QuantumCircuit``.  Other types raise ``TypeError``.
        backend: Reserved for future backend-aware optimization.  Ignored in v0.
        strategy: Optimization strategy.  Currently only ``"geodesic"`` is
            supported.
        return_metadata: When True, return an :class:`OptimizationResult`
            containing the circuit plus statistics.  When False (default),
            return the optimized circuit directly.

    Returns:
        An optimized ``QuantumCircuit``, or an :class:`OptimizationResult`
        when ``return_metadata=True``.

    Raises:
        TypeError: If *circuit* is not a ``QuantumCircuit``.
        ValueError: If *strategy* is not a recognised strategy name.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"optimize() expects a Qiskit QuantumCircuit, got {type(circuit).__name__!r}."
        )
    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Supported strategies: {sorted(_SUPPORTED_STRATEGIES)}."
        )

    from .fusion import fuse_circuit
    from .metrics import gate_count
    from .qiskit_adapter import build_optimized_circuit

    original_count = gate_count(circuit)
    segments, fused_runs = fuse_circuit(circuit)
    optimized = build_optimized_circuit(circuit, segments)
    optimized_count = gate_count(optimized)

    if return_metadata:
        notes: list[str] = []
        if backend is not None:
            notes.append("backend argument is reserved for future use and was ignored.")
        return OptimizationResult(
            circuit=optimized,
            original_gate_count=original_count,
            optimized_gate_count=optimized_count,
            fused_runs=fused_runs,
            strategy=strategy,
            notes=notes,
        )

    return optimized


def summarize_optimization(
    original: QuantumCircuit,
    optimized: QuantumCircuit,
) -> dict[str, Any]:
    """Return a summary dictionary comparing *original* and *optimized*.

    Args:
        original: The input circuit before optimization.
        optimized: The circuit returned by :func:`optimize`.

    Returns:
        Dictionary with keys:
        - ``original_gate_count``
        - ``optimized_gate_count``
        - ``gate_reduction``
        - ``reduction_percent``
    """
    from .metrics import gate_count

    orig = gate_count(original)
    opt = gate_count(optimized)
    reduction = orig - opt
    pct = (reduction / orig * 100.0) if orig > 0 else 0.0

    return {
        "original_gate_count": orig,
        "optimized_gate_count": opt,
        "gate_reduction": reduction,
        "reduction_percent": round(pct, 2),
    }
