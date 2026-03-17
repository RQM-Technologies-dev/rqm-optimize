"""tests/test_optimize_qiskit.py — Integration tests comparing unitaries up to
global phase and verifying optimization metadata.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from rqm_optimize import OptimizationResult, optimize


def equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True if unitary matrices *a* and *b* are equal up to global phase.

    Uses the Hilbert-Schmidt inner product: if a = e^{iα} b then
    |trace(a† b)| = n (matrix dimension), regardless of angle wrapping.
    """
    n = a.shape[0]
    inner = np.trace(a.conj().T @ b)
    return bool(abs(abs(inner) - n) < atol * n)


# ---------------------------------------------------------------------------
# Basic rotation run compression
# ---------------------------------------------------------------------------

def test_rx_ry_rz_run_compressed() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)

    result = optimize(qc, return_metadata=True)
    assert isinstance(result, OptimizationResult)

    orig_u = Operator(qc).data
    opt_u = Operator(result.circuit).data
    assert equal_up_to_phase(orig_u, opt_u)

    assert result.optimized_gate_count <= result.original_gate_count
    assert result.fused_runs >= 1


def test_return_metadata_false_returns_circuit() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.3, 0)
    opt = optimize(qc)
    assert isinstance(opt, QuantumCircuit)


# ---------------------------------------------------------------------------
# Mixed circuit (H + CX + rotations)
# ---------------------------------------------------------------------------

def test_mixed_circuit_equivalence() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.rx(0.4, 1)

    result = optimize(qc, return_metadata=True)

    orig_u = Operator(qc).data
    opt_u = Operator(result.circuit).data
    assert equal_up_to_phase(orig_u, opt_u)

    assert result.optimized_gate_count <= result.original_gate_count


# ---------------------------------------------------------------------------
# Circuit with measurements — check structure preservation
# ---------------------------------------------------------------------------

def test_measurement_circuit_preserves_measurements() -> None:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(0.5, 0)
    qc.measure(0, 0)

    opt = optimize(qc)

    meas_count = sum(1 for instr in opt.data if instr.operation.name == "measure")
    assert meas_count == 1

    # The measure must come last.
    names = [instr.operation.name for instr in opt.data]
    assert names[-1] == "measure"


def test_measurement_count_unchanged() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    opt = optimize(qc)

    orig_meas = sum(1 for i in qc.data if i.operation.name == "measure")
    opt_meas = sum(1 for i in opt.data if i.operation.name == "measure")
    assert orig_meas == opt_meas


# ---------------------------------------------------------------------------
# Identity circuit — zero-gate circuit stays valid
# ---------------------------------------------------------------------------

def test_empty_circuit() -> None:
    qc = QuantumCircuit(2)
    opt = optimize(qc)
    assert opt.num_qubits == 2
    assert len(opt.data) == 0


# ---------------------------------------------------------------------------
# Single-qubit identity chain collapses
# ---------------------------------------------------------------------------

def test_identity_chain_collapses() -> None:
    """Four Rx(0) gates should fuse to something equivalent to identity."""
    qc = QuantumCircuit(1)
    for _ in range(4):
        qc.rx(0.0, 0)

    opt = optimize(qc)

    orig_u = Operator(qc).data
    opt_u = Operator(opt).data
    assert equal_up_to_phase(orig_u, opt_u)
    assert len(opt.data) <= len(qc.data)


# ---------------------------------------------------------------------------
# Multi-rotation deep run
# ---------------------------------------------------------------------------

def test_deep_single_qubit_run() -> None:
    qc = QuantumCircuit(1)
    angles = [0.1 * i for i in range(1, 9)]
    for angle in angles:
        qc.rx(angle, 0)
        qc.rz(angle * 0.5, 0)

    result = optimize(qc, return_metadata=True)

    orig_u = Operator(qc).data
    opt_u = Operator(result.circuit).data
    assert equal_up_to_phase(orig_u, opt_u)

    assert result.optimized_gate_count <= result.original_gate_count


# ---------------------------------------------------------------------------
# metadata fields (including new depth and 1q gate count fields)
# ---------------------------------------------------------------------------

def test_optimization_result_fields() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)

    result = optimize(qc, return_metadata=True)

    assert isinstance(result.circuit, QuantumCircuit)
    assert isinstance(result.original_gate_count, int)
    assert isinstance(result.optimized_gate_count, int)
    assert isinstance(result.original_depth, int)
    assert isinstance(result.optimized_depth, int)
    assert isinstance(result.original_1q_gate_count, int)
    assert isinstance(result.optimized_1q_gate_count, int)
    assert isinstance(result.fused_runs, int)
    assert result.strategy == "geodesic"
    assert result.native_basis is None
    assert isinstance(result.notes, list)


def test_depth_fields_are_correct() -> None:
    """Depth of a linear single-qubit chain must equal gate count."""
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)

    result = optimize(qc, return_metadata=True)

    # Original has 3 sequential single-qubit gates → depth 3.
    assert result.original_depth == 3
    assert result.original_1q_gate_count == 3

    # Optimized fuses to 1 gate.
    assert result.optimized_depth == 1
    assert result.optimized_1q_gate_count == 1

    # Depth is always ≤ original.
    assert result.optimized_depth <= result.original_depth


def test_depth_fields_with_multi_qubit_gate() -> None:
    """CX creates depth; verify original_depth and optimized_depth are sane."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.rx(0.4, 1)

    result = optimize(qc, return_metadata=True)

    # Original: H, CX, Rz, Rx — depth 4 (serial on one path).
    assert result.original_depth >= 3
    assert result.optimized_depth >= 1
    assert result.optimized_depth <= result.original_depth


def test_1q_gate_count_excludes_cx() -> None:
    qc = QuantumCircuit(2)
    qc.rx(0.5, 0)
    qc.cx(0, 1)
    qc.ry(0.3, 1)

    result = optimize(qc, return_metadata=True)

    # Original has 2 single-qubit gates (Rx + Ry); CX is 2-qubit.
    assert result.original_1q_gate_count == 2


# ---------------------------------------------------------------------------
# native_basis parameter
# ---------------------------------------------------------------------------

def test_native_basis_none_is_default() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)

    result = optimize(qc, native_basis=None, return_metadata=True)
    assert result.native_basis is None

    # Default basis produces a single U gate for a 3-gate run.
    gate_names = [instr.operation.name for instr in result.circuit.data]
    assert gate_names == ["u"]


def test_native_basis_ibm_uses_rz_sx() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)
    qc.h(0)
    qc.s(0)

    result = optimize(qc, native_basis="ibm", return_metadata=True)

    assert result.native_basis == "ibm"

    # Verify equivalence.
    orig_u = Operator(qc).data
    opt_u = Operator(result.circuit).data
    assert equal_up_to_phase(orig_u, opt_u)

    # Output gates should only be rz and sx (IBM native set).
    gate_names = {instr.operation.name for instr in result.circuit.data}
    assert gate_names <= {"rz", "sx", "x"}


def test_native_basis_zyz_uses_rz_ry() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)
    qc.rz(0.2, 0)

    result = optimize(qc, native_basis="zyz", return_metadata=True)

    assert result.native_basis == "zyz"

    orig_u = Operator(qc).data
    opt_u = Operator(result.circuit).data
    assert equal_up_to_phase(orig_u, opt_u)

    gate_names = {instr.operation.name for instr in result.circuit.data}
    assert gate_names <= {"rz", "ry"}


def test_native_basis_invalid_raises() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.1, 0)
    with pytest.raises(ValueError, match="native_basis"):
        optimize(qc, native_basis="not_a_basis")


def test_native_basis_preserved_in_result_fields() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(0.3, 0)

    result_ibm = optimize(qc, native_basis="ibm", return_metadata=True)
    result_zyz = optimize(qc, native_basis="zyz", return_metadata=True)

    assert result_ibm.native_basis == "ibm"
    assert result_zyz.native_basis == "zyz"


# ---------------------------------------------------------------------------
# Backend argument note
# ---------------------------------------------------------------------------

def test_backend_note_added_when_provided() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.1, 0)
    result = optimize(qc, backend="fake_backend", return_metadata=True)
    assert any("backend" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output() -> None:
    qc = QuantumCircuit(1)
    qc.rx(0.7, 0)
    qc.ry(0.4, 0)
    qc.rz(0.2, 0)

    opt1 = optimize(qc)
    opt2 = optimize(qc)

    u1 = Operator(opt1).data
    u2 = Operator(opt2).data
    assert np.allclose(u1, u2)

