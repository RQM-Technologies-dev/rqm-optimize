"""tests/test_axis_aware.py — Tests for axis-aware single-qubit compression.

Validates that fused same-axis rotation runs emit named rx/ry/rz gates
(rather than a generic U gate) and that the resulting circuits are equivalent
to their inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from rqm_optimize import optimize
from rqm_optimize.geometry import axis_aligned_rotation, quaternion_canonicalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    n = a.shape[0]
    inner = np.trace(a.conj().T @ b)
    return bool(abs(abs(inner) - n) < atol * n)


# ---------------------------------------------------------------------------
# Unit tests for axis_aligned_rotation
# ---------------------------------------------------------------------------

class TestAxisAlignedRotation:
    def _quat_rz(self, theta: float) -> np.ndarray:
        return quaternion_canonicalize(
            np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])
        )

    def _quat_rx(self, theta: float) -> np.ndarray:
        return quaternion_canonicalize(
            np.array([math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0])
        )

    def _quat_ry(self, theta: float) -> np.ndarray:
        return quaternion_canonicalize(
            np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])
        )

    @pytest.mark.parametrize("theta", [0.3, 0.7, math.pi / 2, math.pi * 0.9])
    def test_rz_detected(self, theta: float) -> None:
        q = self._quat_rz(theta)
        result = axis_aligned_rotation(q)
        assert result is not None
        name, angle = result
        assert name == "z"
        assert abs(angle - theta) < 1e-9

    @pytest.mark.parametrize("theta", [0.4, 1.1, math.pi / 3])
    def test_rx_detected(self, theta: float) -> None:
        q = self._quat_rx(theta)
        result = axis_aligned_rotation(q)
        assert result is not None
        name, angle = result
        assert name == "x"
        assert abs(angle - theta) < 1e-9

    @pytest.mark.parametrize("theta", [0.5, 0.9, math.pi / 4])
    def test_ry_detected(self, theta: float) -> None:
        q = self._quat_ry(theta)
        result = axis_aligned_rotation(q)
        assert result is not None
        name, angle = result
        assert name == "y"
        assert abs(angle - theta) < 1e-9

    def test_generic_rotation_returns_none(self) -> None:
        # A rotation about an off-axis direction must return None.
        q = quaternion_canonicalize(
            np.array([math.cos(0.3), math.sin(0.3) / math.sqrt(2), math.sin(0.3) / math.sqrt(2), 0.0])
        )
        assert axis_aligned_rotation(q) is None

    def test_near_identity_returns_none(self) -> None:
        q = quaternion_canonicalize(np.array([1.0 - 1e-9, 0.0, 0.0, 1e-9]))
        result = axis_aligned_rotation(q)
        # Near-identity: theta ≈ 0, should return None.
        assert result is None

    def test_custom_atol(self) -> None:
        # A slightly off-z axis that passes with a looser tolerance.
        # axis[x] ≈ epsilon / sin(0.5) ≈ epsilon / 0.479; choose epsilon so that
        # axis[x] < 1e-3 (loose) but > 1e-6 (strict).
        epsilon = 1e-4  # axis[x] ≈ 2e-4 — inside 1e-3, outside 1e-6
        q = quaternion_canonicalize(
            np.array([math.cos(0.5), epsilon, 0.0, math.sin(0.5)])
        )
        # Strict tolerance: not aligned.
        assert axis_aligned_rotation(q, atol=1e-6) is None
        # Loose tolerance: aligned.
        result = axis_aligned_rotation(q, atol=1e-3)
        assert result is not None
        assert result[0] == "z"


# ---------------------------------------------------------------------------
# Integration: optimize() emits named gates for axis-aligned runs
# ---------------------------------------------------------------------------

class TestAxisAwareCompression:
    @pytest.mark.parametrize("a,b", [(0.3, 0.5), (0.7, 1.1), (math.pi / 4, math.pi / 4)])
    def test_rz_run_emits_rz(self, a: float, b: float) -> None:
        """Two consecutive Rz gates fuse into a single rz gate."""
        qc = QuantumCircuit(1)
        qc.rz(a, 0)
        qc.rz(b, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["rz"], f"Expected ['rz'] but got {gate_names}"
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    @pytest.mark.parametrize("a,b", [(0.3, 0.5), (0.6, 0.9)])
    def test_rx_run_emits_rx(self, a: float, b: float) -> None:
        """Two consecutive Rx gates fuse into a single rx gate."""
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["rx"], f"Expected ['rx'] but got {gate_names}"
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    @pytest.mark.parametrize("a,b", [(0.4, 0.6), (1.0, 0.5)])
    def test_ry_run_emits_ry(self, a: float, b: float) -> None:
        """Two consecutive Ry gates fuse into a single ry gate."""
        qc = QuantumCircuit(1)
        qc.ry(a, 0)
        qc.ry(b, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["ry"], f"Expected ['ry'] but got {gate_names}"
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    def test_rz_three_gates_emits_rz(self) -> None:
        """Three consecutive Rz gates fuse into a single rz gate."""
        qc = QuantumCircuit(1)
        qc.rz(0.3, 0)
        qc.rz(0.5, 0)
        qc.rz(0.2, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["rz"]
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    def test_rz_angle_matches_sum(self) -> None:
        """The fused rz angle equals the sum of the individual angles."""
        a, b = 0.4, 0.6
        qc = QuantumCircuit(1)
        qc.rz(a, 0)
        qc.rz(b, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["rz"]
        emitted_angle = float(opt.data[0].operation.params[0])
        # Angle should equal a + b (modulo 2π wrapping inside arccos, but close).
        assert abs(emitted_angle - (a + b)) < 1e-6

    def test_mixed_axes_still_emits_u(self) -> None:
        """A run of mixed-axis gates (non-axis-aligned result) still emits U."""
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        qc.ry(0.3, 0)
        qc.rz(0.2, 0)

        opt = optimize(qc)

        gate_names = [instr.operation.name for instr in opt.data]
        assert gate_names == ["u"]
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    def test_rz_run_with_cx_boundary(self) -> None:
        """Rz gates on separate sides of a CX are each compressed independently."""
        qc = QuantumCircuit(2)
        qc.rz(0.3, 0)
        qc.rz(0.5, 0)
        qc.cx(0, 1)
        qc.rz(0.4, 1)
        qc.rz(0.6, 1)

        opt = optimize(qc)

        rz_ops = [instr for instr in opt.data if instr.operation.name == "rz"]
        cx_ops = [instr for instr in opt.data if instr.operation.name == "cx"]
        assert len(rz_ops) == 2
        assert len(cx_ops) == 1
        assert equal_up_to_phase(Operator(qc).data, Operator(opt).data)

    def test_axis_aware_with_metadata(self) -> None:
        """OptimizationResult reflects the gate reduction from axis-aware compression."""
        from rqm_optimize import OptimizationResult

        qc = QuantumCircuit(1)
        qc.rz(0.3, 0)
        qc.rz(0.4, 0)
        qc.rz(0.5, 0)

        result = optimize(qc, return_metadata=True)
        assert isinstance(result, OptimizationResult)
        assert result.original_1q_gate_count == 3
        assert result.optimized_1q_gate_count == 1
        assert result.fused_runs == 1

    def test_axis_aware_produces_fewer_gates_than_u_for_ibm_basis(self) -> None:
        """For axis-aligned runs, axis-aware (1 gate) wins over ZSX (≥1 gate)."""
        qc = QuantumCircuit(1)
        qc.rz(0.4, 0)
        qc.rz(0.5, 0)

        # With default basis: should emit 1 rz via axis-aware.
        opt_default = optimize(qc)
        names_default = [i.operation.name for i in opt_default.data]
        assert names_default == ["rz"]

        # With IBM basis: axis-aware still fires first and emits 1 rz.
        opt_ibm = optimize(qc, native_basis="ibm")
        names_ibm = [i.operation.name for i in opt_ibm.data]
        assert names_ibm == ["rz"]

        # Both must be equivalent to the original.
        assert equal_up_to_phase(Operator(qc).data, Operator(opt_default).data)
        assert equal_up_to_phase(Operator(qc).data, Operator(opt_ibm).data)
