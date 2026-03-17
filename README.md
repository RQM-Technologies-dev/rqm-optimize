# rqm-optimize

> `rqm-optimize` is the SU(2)-aware optimization layer for the RQM ecosystem. It compresses contiguous single-qubit gate runs into shorter equivalent forms, reducing unnecessary depth while preserving circuit behavior up to global phase.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Purpose

RQM is not only a representation layer — it is a **performance layer**.

`rqm-optimize` accepts a Qiskit `QuantumCircuit`, scans it for contiguous single-qubit gate runs, fuses those runs into minimal SU(2)-equivalent operations, and returns a simplified circuit that is unitary-equivalent to the original up to global phase.

---

## Stack placement

```text
rqm-core      → canonical quaternion / SU(2) / Bloch math
rqm-compiler  → circuit construction / normalization
rqm-qiskit    → Qiskit lowering / execution bridge
rqm-braket    → Braket lowering / execution bridge
rqm-optimize  → optimization / compression layer  ← this package
```

The intended end-to-end workflow is:

```text
construct (rqm-compiler) → lower (rqm-qiskit) → optimize (rqm-optimize) → run
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
qc.h(0)
qc.s(0)
qc.t(0)

result = optimize(qc, return_metadata=True)

print("original gates:", result.original_gate_count)    # 6
print("optimized gates:", result.optimized_gate_count)  # 1
print("fused runs:", result.fused_runs)                 # 1
print("original depth:", result.original_depth)         # 6
print("optimized depth:", result.optimized_depth)       # 1
print(result.circuit)
```

### Compiler-path integration

If you are using `rqm-compiler` to construct circuits and `rqm-qiskit` to lower them to Qiskit, pass the lowered circuit directly to `optimize`:

```python
# construct → lower → optimize → run
from rqm_qiskit import to_qiskit       # rqm-qiskit lowering bridge
from rqm_optimize import optimize

qiskit_circuit = to_qiskit(compiled_circuit)   # your rqm-compiler output
optimized = optimize(qiskit_circuit)
# submit optimized to your backend of choice
```

### Native-basis preference

Request IBM-native decomposition (`rz` + `sx`) to produce circuits that map directly to common superconducting hardware gate sets:

```python
result = optimize(qc, native_basis="ibm", return_metadata=True)
# output gates are rz and sx — no transpilation step needed for IBM backends
```

Supported `native_basis` values:

| Value | Decomposition | Gates |
|-------|--------------|-------|
| `None` (default) | Compact U basis | `u` |
| `"ibm"` | IBM hardware native | `rz`, `sx` |
| `"zyz"` | Analytic Euler | `rz`, `ry` |

---

## What v0.1 does

- Detects **contiguous single-qubit gate runs** on each qubit.
- Fuses each run into a **single SU(2)-equivalent gate** using matrix
  multiplication followed by Qiskit's `OneQubitEulerDecomposer`.
- Supports **native-basis preference** so fused runs can be emitted directly
  as IBM-native (`rz`/`sx`) or analytic ZYZ gates.
- Skips fusion when the decomposition would produce more gates than the
  original (i.e., only applies optimizations that reduce or maintain gate count).
- Preserves **barriers, measurements, resets, and multi-qubit gates** exactly
  as hard boundaries.
- Never mutates the input circuit.
- Returns deterministic output.
- Reports rich metadata: total gate count, circuit depth, single-qubit gate
  count, fused run count — both before and after.

### Supported gates in v0.1

`rx`, `ry`, `rz`, `u`, `u3`, `u2`, `u1`, `p`, `x`, `y`, `z`, `h`, `s`, `sdg`,
`t`, `tdg`, `id`, `sx`, `sxdg`, `r`, and any generic single-qubit
`UnitaryGate` whose matrix can be extracted.

---

## What v0.1 does not yet do

- Backend-aware native-axis alignment using calibration data (planned for v0.2).
- Quaternionic error metrics and drift-aware path selection (planned).
- Braket circuit support (planned).
- Two-qubit gate optimization.

---

## `OptimizationResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `circuit` | `QuantumCircuit` | The optimized circuit |
| `original_gate_count` | `int` | Total gate count before optimization |
| `optimized_gate_count` | `int` | Total gate count after optimization |
| `original_depth` | `int` | Circuit depth before optimization |
| `optimized_depth` | `int` | Circuit depth after optimization |
| `original_1q_gate_count` | `int` | Single-qubit gate count before |
| `optimized_1q_gate_count` | `int` | Single-qubit gate count after |
| `fused_runs` | `int` | Number of runs fused (≥ 2 gates → 1) |
| `strategy` | `str` | Optimization strategy used |
| `native_basis` | `str \| None` | Decomposition basis preference |
| `notes` | `list[str]` | Human-readable optimization notes |

---

## Architecture

```
src/rqm_optimize/
├── __init__.py         # Public API: optimize, OptimizationResult
├── optimizer.py        # Type dispatch, strategy validation, result packaging
├── fusion.py           # Single-qubit run identification and matrix fusion
├── geometry.py         # SU(2) / global-phase normalization helpers
├── metrics.py          # Gate count, depth, 1q gate count, matrix error norms
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
- Metadata fields (`original_depth`, `optimized_depth`, `original_1q_gate_count`, `optimized_1q_gate_count`) and determinism.
- `native_basis` parameter: IBM (`rz`/`sx`) and ZYZ decomposition paths.

---

## Product ladder

```text
rqm-optimize     → improves circuits today
rqm-calibration  → backend / drift / native-axis intelligence  (future)
rqm-noise        → quaternionic noise and error modeling        (future)
```

---

## License

MIT © RQM Technologies
