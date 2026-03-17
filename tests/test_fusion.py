"""tests/test_fusion.py — Tests for single-qubit run identification and fusion."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from rqm_optimize import optimize


def unitaries_equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True if *a* and *b* are equal up to a global phase.

    Uses the Hilbert-Schmidt inner product: if a = e^{iα} b then
    |trace(a† b)| = n (matrix dimension), regardless of angle wrapping.
    """
    n = a.shape[0]
    inner = np.trace(a.conj().T @ b)
    return bool(abs(abs(inner) - n) < atol * n)


# ---------------------------------------------------------------------------
# Test 1 — three rotations on one qubit compress to a simpler run
# ---------------------------------------------------------------------------

def test_three_rotations_are_fused() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)

    opt = optimize(qc)

    # Optimized must have fewer or equal gates.
    assert len(opt.data) <= len(qc.data)

    # Must be equivalent up to global phase.
    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert unitaries_equal_up_to_phase(orig_u, opt_u)


# ---------------------------------------------------------------------------
# Test 2 — a CX gate splits the single-qubit runs correctly
# ---------------------------------------------------------------------------

def test_cx_splits_single_qubit_runs() -> None:
    qc = QuantumCircuit(2)
    qc.rx(0.4, 0)
    qc.ry(0.2, 0)
    qc.cx(0, 1)
    qc.rz(0.5, 1)
    qc.rx(0.1, 1)

    opt = optimize(qc)

    # The CX must still be present.
    cx_ops = [instr for instr in opt.data if instr.operation.name == "cx"]
    assert len(cx_ops) == 1

    # Equivalence up to global phase for the unitary part.
    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert unitaries_equal_up_to_phase(orig_u, opt_u)


# ---------------------------------------------------------------------------
# Test 3 — barriers stop fusion
# ---------------------------------------------------------------------------

def test_barriers_stop_fusion() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.4, 0)
    qc.barrier(0)
    qc.ry(0.3, 0)

    opt = optimize(qc)

    # Barrier must still be present.
    barriers = [instr for instr in opt.data if instr.operation.name == "barrier"]
    assert len(barriers) == 1

    # Gates before and after barrier must not have been fused together.
    # The rx and ry cannot be merged across the barrier; check gate ordering.
    gate_names = [instr.operation.name for instr in opt.data]
    barrier_idx = gate_names.index("barrier")
    assert barrier_idx > 0, "There must be gates before the barrier."
    assert barrier_idx < len(gate_names) - 1, "There must be gates after the barrier."

    # Both sides must still implement the correct unitary for circuits
    # where the barrier doesn't split the qubit count — use Operator.
    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert unitaries_equal_up_to_phase(orig_u, opt_u)


# ---------------------------------------------------------------------------
# Test 4 — measurements stop fusion
# ---------------------------------------------------------------------------

def test_measurements_stop_fusion() -> None:
    qc = QuantumCircuit(1, 1)
    qc.rx(0.4, 0)
    qc.measure(0, 0)
    qc.ry(0.3, 0)  # After measurement — not fuseable with before.

    opt = optimize(qc)

    # Measurement must still be present.
    measurements = [instr for instr in opt.data if instr.operation.name == "measure"]
    assert len(measurements) == 1

    # Gate order: gates, then measure, then gate.
    names = [instr.operation.name for instr in opt.data]
    meas_idx = names.index("measure")
    assert meas_idx > 0
    assert meas_idx < len(names) - 1


# ---------------------------------------------------------------------------
# Test 5 — a single one-qubit gate is preserved unchanged
# ---------------------------------------------------------------------------

def test_single_gate_preserved() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)

    opt = optimize(qc)

    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert unitaries_equal_up_to_phase(orig_u, opt_u)


# ---------------------------------------------------------------------------
# Test 6 — generic UnitaryGate is preserved equivalently
# ---------------------------------------------------------------------------

def test_generic_unitary_gate_preserved() -> None:
    from qiskit.circuit.library import UnitaryGate

    # Build a random SU(2) matrix.
    theta = math.pi / 5
    mat = np.array(
        [
            [math.cos(theta / 2), -math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=complex,
    )
    qc = QuantumCircuit(1)
    qc.append(UnitaryGate(mat), [0])
    qc.rx(0.3, 0)

    opt = optimize(qc)

    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert unitaries_equal_up_to_phase(orig_u, opt_u)


# ---------------------------------------------------------------------------
# Test 7 — original circuit is never mutated
# ---------------------------------------------------------------------------

def test_original_not_mutated() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    original_data_len = len(qc.data)

    optimize(qc)

    assert len(qc.data) == original_data_len
