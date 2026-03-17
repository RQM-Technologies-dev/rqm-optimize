# rqm-optimize

> `rqm-optimize` improves quantum circuits by compressing single-qubit gate runs through SU(2)-aware fusion, reducing unnecessary gate depth while preserving circuit behavior.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Purpose

`rqm-optimize` is the **optimization layer** of the RQM Technologies quantum software stack. It accepts a Qiskit `QuantumCircuit`, scans it for contiguous single-qubit gate runs, fuses those runs into minimal SU(2)-equivalent operations, and returns a simplified circuit that is unitary-equivalent to the original up to global phase.

---

## Stack placement

```text
rqm-core      → canonical quaternion / SU(2) / Bloch math
rqm-compiler  → circuit construction / normalization
rqm-qiskit    → Qiskit lowering / execution bridge
rqm-braket    → Braket lowering / execution bridge
rqm-optimize  → optimization / compression layer  ← this package
```

---

## Installation

```bash
pip install rqm-optimize
```

Or from source:

```bash
git clone https://github.com/RQM-Technologies-dev/rqm-optimize.git
cd rqm-optimize
pip install -e ".[dev]"
```

---

## Quickstart

```python
from qiskit import QuantumCircuit
from rqm_optimize import optimize

qc = QuantumCircuit(1)
qc.rx(0.5, 0)
qc.ry(0.3, 0)
qc.rz(0.2, 0)

# Simple usage — returns an optimized QuantumCircuit.
qc_opt = optimize(qc)

# With metadata.
result = optimize(qc, return_metadata=True)
print(result.original_gate_count)   # 3
print(result.optimized_gate_count)  # 1
print(result.fused_runs)            # 1
print(result.circuit)
```

---

## What v0 does

- Detects **contiguous single-qubit gate runs** on each qubit.
- Fuses each run into a **single SU(2)-equivalent gate** using matrix
  multiplication followed by Qiskit's `OneQubitEulerDecomposer`.
- Skips fusion when the decomposition would produce more gates than the
  original (i.e., only applies optimizations that reduce or maintain gate count).
- Preserves **barriers, measurements, resets, and multi-qubit gates** exactly
  as hard boundaries.
- Never mutates the input circuit.
- Returns deterministic output.

### Supported gates in v0

`rx`, `ry`, `rz`, `u`, `u3`, `u2`, `u1`, `p`, `x`, `y`, `z`, `h`, `s`, `sdg`,
`t`, `tdg`, `id`, `sx`, `sxdg`, `r`, and any generic single-qubit
`UnitaryGate` whose matrix can be extracted.

---

## What v0 does not yet do

- Backend-aware native-axis alignment (planned for v0.2).
- Quaternionic error metrics and drift-aware path selection (planned).
- Braket circuit support (planned).
- Two-qubit gate optimization.

---

## Architecture

```
src/rqm_optimize/
├── __init__.py         # Public API: optimize, OptimizationResult
├── optimizer.py        # Type dispatch, strategy validation, result packaging
├── fusion.py           # Single-qubit run identification and matrix fusion
├── geometry.py         # SU(2) / global-phase normalization helpers
├── metrics.py          # Gate count metrics, matrix error norms
├── qiskit_adapter.py   # Qiskit instruction inspection, matrix extraction, Euler emission
└── py.typed            # PEP 561 marker
```

The public surface area is intentionally minimal: `optimize()` and
`OptimizationResult`. All internal helpers are private.

---

## Development and testing

```bash
# Install with dev dependencies.
pip install -e ".[dev]"

# Run tests.
pytest

# Run the example.
python examples/basic_optimize.py
```

Tests cover:

- Public API importability and `__all__` contract.
- Fusion correctness (runs compressed, boundaries respected, equivalence up to global phase).
- Integration tests comparing unitaries using `qiskit.quantum_info.Operator`.
- Measurement / barrier / multi-qubit structure preservation.
- Metadata fields and determinism.

---

## License

MIT © RQM Technologies
