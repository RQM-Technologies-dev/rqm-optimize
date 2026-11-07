"""Exact composable contract including scalar phase and small gates."""

from io import BytesIO
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from rqm_optimize import optimize


@pytest.mark.parametrize("basis", [None, "zyz", "ibm"])
@pytest.mark.parametrize("family", ["rx", "ry", "rz", "mixed", "tiny", "phase", "adverse"])
def test_exact_composition_inverse_control_and_qpy(basis, family):
    circuit = QuantumCircuit(2)
    circuit.global_phase = 0.37
    if family in ("rx", "ry", "rz"):
        getattr(circuit, family)(2 * math.pi, 0)
        getattr(circuit, family)(0.0, 0)
    elif family == "tiny":
        circuit.rx(1.5e-8, 0)
        circuit.rz(1e-12, 0)
    elif family == "phase":
        circuit.p(1e-12, 0)
        circuit.p(1e-12, 0)
    elif family == "adverse":
        circuit.rx(0.3, 0)
        circuit.ry(1e-7, 0)
        circuit.rz(-0.4, 0)
    else:
        circuit.h(0)
        circuit.sx(0)
        circuit.x(0)
        circuit.y(0)
    circuit.cx(0, 1)
    circuit.sxdg(1)
    circuit.t(1)
    after = optimize(circuit, native_basis=basis)
    stream = BytesIO()
    qpy.dump(after, stream)
    stream.seek(0)
    after = qpy.load(stream)[0]
    for before, result in [
        (circuit, after),
        (circuit.inverse(), after.inverse()),
        (circuit.to_gate().control(), after.to_gate().control()),
        (circuit.compose(circuit), after.compose(after)),
    ]:
        a, b = Operator(before).data, Operator(result).data
        assert np.linalg.norm(a - b) / np.sqrt(a.shape[0]) < 1e-10


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("angle", [1e-12, 2 * math.pi, -2 * math.pi, 3 * math.pi])
def test_signed_axis_against_independent_pauli_exponential(axis, angle):
    """Tight tolerance detects dropping even the smallest tested rotation."""
    paulis = {
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.diag([1, -1]),
    }
    circuit = QuantumCircuit(1)
    getattr(circuit, "r" + axis)(angle, 0)
    getattr(circuit, "r" + axis)(0.0, 0)
    expected = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * paulis[axis]
    actual = Operator(optimize(circuit)).data
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
