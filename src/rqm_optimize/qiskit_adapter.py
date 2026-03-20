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

# Gate names that unambiguously identify an IBM-style hardware backend.
# Both must be present in the backend's reported basis gate set.
_IBM_INDICATOR_GATES = frozenset({"sx", "rz"})


def infer_native_basis_from_backend(backend: object) -> str | None:
    """Inspect *backend* and return a suitable ``native_basis`` string, or ``None``.

    Heuristic:
    - If the backend advertises ``sx`` and ``rz`` in its native gate set,
      it is treated as IBM-style and ``"ibm"`` is returned.
    - In all other cases — including backends whose gate set cannot be
      determined — ``None`` is returned so the caller falls back to the
      default compact ``"U"`` basis.

    This function never raises: all backend API calls are guarded by a broad
    ``except`` clause so that an unusual or mock backend cannot break
    optimization.

    Args:
        backend: Any object passed as the ``backend`` argument to
            :func:`~rqm_optimize.optimize`.

    Returns:
        ``"ibm"`` when the backend looks IBM-style, ``None`` otherwise.
    """
    gate_names: set[str] = set()

    # Qiskit 1.x: BackendV2 exposes operation_names directly.
    try:
        names = backend.operation_names  # type: ignore[union-attr]
        if names is not None:
            gate_names.update(str(n) for n in names)
    except Exception:  # noqa: BLE001
        pass

    # Qiskit 0.x: BackendV1 exposes configuration().basis_gates.
    if not gate_names:
        try:
            cfg = backend.configuration()  # type: ignore[union-attr]
            basis = cfg.basis_gates
            if basis is not None:
                gate_names.update(str(n) for n in basis)
        except Exception:  # noqa: BLE001
            pass

    if not gate_names:
        return None

    if _IBM_INDICATOR_GATES <= gate_names:
        return "ibm"

    return None


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


def emit_axis_aligned_gate(
    matrix: np.ndarray,
    circuit: "QuantumCircuit",
    qubit: "Qubit",
) -> int:
    """Emit a single Rx/Ry/Rz gate if *matrix* is a Cartesian-axis rotation.

    Inspects the quaternion representation of *matrix* to determine whether the
    fused rotation is aligned with the x, y, or z axis on S³.  When it is, a
    single named rotation gate (``rx``, ``ry``, or ``rz``) is appended to
    *circuit* and 1 is returned.  When the rotation is generic or near-identity,
    the circuit is left unchanged and 0 is returned.

    This is the axis-aware compression pass: it operates directly on the S³
    quaternion representation rather than on matrices, and produces a semantically
    named gate that is directly executable on hardware (in particular, ``rz`` is
    virtual/"free" on many backends).

    Args:
        matrix: 2×2 SU(2)-normalized unitary matrix for the fused run.
        circuit: Target ``QuantumCircuit`` to append the gate to.
        qubit: The target qubit.

    Returns:
        1 if an axis-aligned gate was emitted, 0 otherwise.
    """
    from .geometry import axis_aligned_rotation, su2_to_quaternion

    q = su2_to_quaternion(matrix)
    result = axis_aligned_rotation(q)
    if result is None:
        return 0
    axis_name, theta = result
    if axis_name == "x":
        circuit.rx(theta, qubit)
    elif axis_name == "y":
        circuit.ry(theta, qubit)
    else:
        circuit.rz(theta, qubit)
    return 1


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
        unitary: 2×2 complex SU(2)-normalized unitary matrix.
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

            from qiskit import QuantumCircuit as QC

            # First pass: axis-aware compression.
            # If the fused rotation is aligned with the x, y, or z axis, emit
            # a single rx/ry/rz gate directly from the S³ quaternion form.
            # This always produces 1 gate (≤ original_count which is ≥ 2).
            tmp = QC(1)
            if emit_axis_aligned_gate(matrix, tmp, tmp.qubits[0]) > 0:
                for instr in tmp.data:
                    out.append(instr.operation, [qubit])
                continue

            # Second pass: Euler decomposer for generic (non-axis-aligned) rotations.
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
