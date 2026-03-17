"""qiskit_adapter.py — Qiskit-specific instruction inspection, matrix
extraction, and circuit emission for the optimizer.

Owns all Qiskit API surface so the rest of the package stays framework-
agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction, Qubit

# Gate names that are unambiguously single-qubit unitaries and whose matrix
# Qiskit always provides without parameters being None.
_KNOWN_SINGLE_QUBIT_GATE_NAMES = frozenset(
    {
        "rx", "ry", "rz",
        "u", "u3", "u2", "u1",
        "p",
        "x", "y", "z",
        "h",
        "s", "sdg",
        "t", "tdg",
        "id",
        "sx", "sxdg",
        # r gate (general single-qubit rotation)
        "r",
    }
)

# Gate classes that are hard boundaries — never fuse across these.
_BOUNDARY_GATE_NAMES = frozenset(
    {
        "barrier",
        "measure",
        "reset",
    }
)

# Map user-facing native_basis names → OneQubitEulerDecomposer basis strings.
# "U" is the default compact basis (single U gate per run).
# "ibm" targets IBM hardware native gates: rz and sx.
# "zyz" produces an analytic Rz-Ry-Rz decomposition.
NATIVE_BASIS_MAP: dict[str, str] = {
    "ibm": "ZSX",
    "zyz": "ZYZ",
}

# The default decomposer basis when no native_basis is requested.
_DEFAULT_DECOMPOSER_BASIS = "U"


def is_fuseable_single_qubit(op: "Instruction") -> bool:
    """Return True if *op* is a single-qubit unitary gate that may be fused.

    An instruction is fuseable when:
    - it acts on exactly one qubit (checked by the caller via ``qargs``),
    - it is not a barrier, measurement, or reset,
    - it is not a classical-conditioned instruction,
    - it is not a control-flow instruction,
    - its 2×2 unitary matrix can be extracted.

    Args:
        op: A Qiskit ``Instruction`` object.

    Returns:
        True when the instruction is safe to fuse.
    """
    from qiskit.circuit import Barrier, Measure, Reset
    from qiskit.circuit.controlflow import ControlFlowOp

    if isinstance(op, (Barrier, Measure, Reset)):
        return False
    if isinstance(op, ControlFlowOp):
        return False
    # Classical conditions make semantics non-unitary.
    if getattr(op, "condition", None) is not None:
        return False
    # num_qubits check: caller also checks qargs length, but be explicit.
    if op.num_qubits != 1:
        return False

    if op.name in _KNOWN_SINGLE_QUBIT_GATE_NAMES:
        return True

    # Try matrix extraction for any other single-qubit gate.
    return _try_extract_matrix(op) is not None


def extract_matrix(op: "Instruction") -> np.ndarray:
    """Return the 2×2 unitary matrix for a single-qubit gate *op*.

    Args:
        op: A fuseable single-qubit Qiskit ``Instruction``.

    Returns:
        2×2 complex numpy array.

    Raises:
        ValueError: If the matrix cannot be extracted.
    """
    mat = _try_extract_matrix(op)
    if mat is None:
        raise ValueError(
            f"Cannot extract matrix from instruction '{op.name}'. "
            "Only fuseable single-qubit unitary gates are supported."
        )
    return mat


def _try_extract_matrix(op: "Instruction") -> np.ndarray | None:
    """Attempt matrix extraction; return None on failure."""
    try:
        mat = op.to_matrix()
        if mat is not None and mat.shape == (2, 2):
            return np.asarray(mat, dtype=complex)
    except Exception:  # noqa: BLE001
        pass
    return None


def emit_euler_gate(
    unitary: np.ndarray,
    circuit: "QuantumCircuit",
    qubit: "Qubit",
    basis: str = _DEFAULT_DECOMPOSER_BASIS,
) -> int:
    """Decompose *unitary* and append the resulting gates onto *circuit*.

    Uses Qiskit's ``OneQubitEulerDecomposer`` with the requested *basis* to
    produce a compact, exact decomposition.

    Args:
        unitary: 2×2 complex SU(2)-normalised unitary matrix.
        circuit: The ``QuantumCircuit`` to append gates to.
        qubit: The target qubit.
        basis: Euler decomposer basis string (e.g. ``"U"``, ``"ZSX"``,
            ``"ZYZ"``).  Defaults to ``"U"`` for the most compact output.

    Returns:
        Number of gates appended.
    """
    from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer

    decomposer = OneQubitEulerDecomposer(basis=basis)
    decomposed = decomposer(unitary)
    before = len(circuit.data)
    for instr in decomposed.data:
        circuit.append(instr.operation, [qubit])
    return len(circuit.data) - before


def build_optimized_circuit(
    original: "QuantumCircuit",
    segments: list[dict],
    basis: str = _DEFAULT_DECOMPOSER_BASIS,
) -> "QuantumCircuit":
    """Reconstruct a circuit from a list of *segments*.

    Each segment is one of:

    - ``{"type": "passthrough", "instruction": CircuitInstruction}`` — copy
      the instruction unchanged.
    - ``{"type": "fused_run", "qubit": Qubit, "matrix": ndarray,
        "original_count": int}`` — emit an Euler decomposition for the fused
        matrix on *qubit*, but only if the decomposition is not worse than the
        original (by gate count).

    Args:
        original: The original ``QuantumCircuit`` (used for register structure).
        segments: Ordered list of segment descriptors.
        basis: Euler decomposer basis string to use for fused runs.

    Returns:
        A new ``QuantumCircuit`` with the optimized instruction sequence.
    """
    from qiskit import QuantumCircuit

    out = QuantumCircuit(*original.qregs, *original.cregs, name=original.name)

    for seg in segments:
        if seg["type"] == "passthrough":
            instr = seg["instruction"]
            out.append(instr.operation, instr.qubits, instr.clbits)
        elif seg["type"] == "fused_run":
            qubit: "Qubit" = seg["qubit"]
            matrix: np.ndarray = seg["matrix"]
            original_count: int = seg["original_count"]

            # Build a temporary single-qubit sub-circuit to count decomposition.
            from qiskit import QuantumCircuit as QC

            tmp = QC(1)
            emitted = emit_euler_gate(matrix, tmp, tmp.qubits[0], basis=basis)

            if emitted <= original_count:
                # Decomposition is equal or better — use it.
                for instr in tmp.data:
                    out.append(instr.operation, [qubit])
            else:
                # Decomposition is worse — keep the original gates.
                for raw_instr in seg["original_instructions"]:
                    out.append(raw_instr.operation, raw_instr.qubits, raw_instr.clbits)
        else:
            raise ValueError(f"Unknown segment type: {seg['type']!r}")

    return out
