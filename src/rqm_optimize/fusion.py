"""fusion.py — Single-qubit run identification and matrix fusion.

Scans a Qiskit circuit qubit-by-qubit, identifies maximal contiguous runs of
fuseable single-qubit gates, computes the combined unitary for each run, and
produces a segment list consumed by ``qiskit_adapter.build_optimized_circuit``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .qiskit_adapter import extract_matrix, is_fuseable_single_qubit

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import CircuitInstruction


def fuse_circuit(circuit: "QuantumCircuit") -> tuple[list[dict], int]:
    """Identify single-qubit runs in *circuit* and return a segment list.

    Scans the circuit in instruction order and groups consecutive fuseable
    single-qubit gates on the same qubit.  Hard boundaries (barriers,
    measurements, resets, multi-qubit gates, conditionals) flush any open run.

    Each open run is fused into a single 2×2 unitary via matrix multiplication.

    Args:
        circuit: The input ``QuantumCircuit``.

    Returns:
        A tuple of:
        - ``segments``: ordered list of segment dicts for
          ``build_optimized_circuit``.
        - ``fused_runs``: number of runs that contained ≥ 2 gates.
    """
    # Map qubit object → index (for per-qubit run tracking).
    qubit_index: dict = {bit: idx for idx, bit in enumerate(circuit.qubits)}

    # Per-qubit accumulator for the current open run.
    # Each entry: list of CircuitInstruction objects.
    open_runs: dict[int, list["CircuitInstruction"]] = {
        i: [] for i in range(circuit.num_qubits)
    }

    # Final ordered segment list.
    segments: list[dict] = []

    def flush_run(q_idx: int) -> None:
        """Flush the open run for qubit *q_idx* into the segment list."""
        run = open_runs[q_idx]
        if not run:
            return
        qubit = circuit.qubits[q_idx]
        if len(run) == 1:
            # Single-gate run — emit as passthrough (no fusion benefit).
            segments.append({"type": "passthrough", "instruction": run[0]})
        else:
            # Fuse the run into one matrix.
            combined = _fuse_matrices(run)
            segments.append(
                {
                    "type": "fused_run",
                    "qubit": qubit,
                    "matrix": combined,
                    "original_count": len(run),
                    "original_instructions": list(run),
                }
            )
        open_runs[q_idx] = []

    def flush_all() -> None:
        for q_idx in range(circuit.num_qubits):
            flush_run(q_idx)

    for instr in circuit.data:
        op = instr.operation
        qargs = instr.qubits

        if len(qargs) == 1:
            q_idx = qubit_index[qargs[0]]
            if is_fuseable_single_qubit(op):
                open_runs[q_idx].append(instr)
            else:
                # Boundary on this qubit.
                flush_run(q_idx)
                segments.append({"type": "passthrough", "instruction": instr})
        else:
            # Multi-qubit instruction — flush all involved qubits.
            for qarg in qargs:
                flush_run(qubit_index[qarg])
            segments.append({"type": "passthrough", "instruction": instr})

    # Flush any remaining open runs at end of circuit.
    flush_all()

    fused_runs = sum(
        1
        for seg in segments
        if seg["type"] == "fused_run" and seg["original_count"] >= 2
    )

    return segments, fused_runs


def _fuse_matrices(run: list["CircuitInstruction"]) -> np.ndarray:
    """Compose exact U(2) gates using core quaternion factors and scalar phase."""
    from rqm_core.quaternion import Quaternion
    from rqm_core.su2 import factor_u2_global_phase, su2_to_quaternion, quaternion_to_su2

    q_accum = Quaternion(1.0, 0.0, 0.0, 0.0)
    phase = 0.0
    for instr in run:
        gate_phase, special = factor_u2_global_phase(extract_matrix(instr.operation))
        phase += gate_phase
        q_accum = (su2_to_quaternion(special) * q_accum).normalize()
    return np.exp(1j * phase) * quaternion_to_su2(q_accum)
