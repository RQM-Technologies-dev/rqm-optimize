"""metrics.py — Gate count and optimization quality metrics.

Provides simple, deterministic metrics for before/after comparisons.
Structured to accept future quaternionic misalignment metrics without
changing the public surface area.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def gate_count(circuit: "QuantumCircuit") -> int:
    """Return the total number of gate instructions in *circuit*.

    Measurements, barriers, and resets are excluded; only proper gate
    operations are counted.

    Args:
        circuit: A Qiskit ``QuantumCircuit``.

    Returns:
        Integer gate count.
    """
    from qiskit.circuit import Barrier, Measure, Reset  # type: ignore[attr-defined]

    count = 0
    for instr in circuit.data:
        op = instr.operation
        if not isinstance(op, (Barrier, Measure, Reset)):
            count += 1
    return count


def single_qubit_run_count(circuit: "QuantumCircuit") -> int:
    """Return the number of contiguous single-qubit gate runs in *circuit*.

    A *run* is a maximal contiguous sequence of single-qubit gates on the
    same qubit that is not interrupted by a barrier, measurement, reset, or
    multi-qubit gate.

    Args:
        circuit: A Qiskit ``QuantumCircuit``.

    Returns:
        Number of single-qubit runs across all qubits.
    """
    from .qiskit_adapter import is_fuseable_single_qubit

    # Track whether each qubit is currently inside a run.
    in_run: dict[int, bool] = {i: False for i in range(circuit.num_qubits)}
    qubit_index = {bit: idx for idx, bit in enumerate(circuit.qubits)}
    run_count = 0

    for instr in circuit.data:
        op = instr.operation
        qargs = instr.qubits
        if len(qargs) == 1:
            q = qubit_index[qargs[0]]
            if is_fuseable_single_qubit(op):
                if not in_run[q]:
                    in_run[q] = True
                    run_count += 1
            else:
                in_run[q] = False
        else:
            for qarg in qargs:
                q = qubit_index[qarg]
                in_run[q] = False

    return run_count


def matrix_error_norm(
    a: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
) -> float:
    """Return the Frobenius norm of ``a − b`` after global-phase alignment.

    Useful for verifying that two unitaries represent the same operation up
    to global phase.

    Args:
        a: First unitary matrix.
        b: Second unitary matrix.

    Returns:
        Float Frobenius distance after removing global phase from both.
    """
    from .geometry import remove_global_phase

    return float(np.linalg.norm(remove_global_phase(a) - remove_global_phase(b)))
