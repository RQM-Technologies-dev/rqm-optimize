# AGENTS.md — rqm-optimize repo discipline

## Purpose

`rqm-optimize` is the **optimization layer** in the RQM Technologies quantum software stack.

Its job is to reduce circuit depth and gate count by compressing single-qubit gate runs through SU(2)-aware fusion, and to provide a foundation for backend-aware optimization in future versions.

---

## Stack position

```text
rqm-core      → canonical quaternion / SU(2) / Bloch math
rqm-compiler  → circuit construction / normalization
rqm-qiskit    → Qiskit lowering / execution bridge
rqm-braket    → Braket lowering / execution bridge
rqm-optimize  → optimization / compression layer  ← this repo
```

---

## What belongs here

- Single-qubit run fusion and compression logic
- SU(2)-aware decomposition helpers that are specific to optimization workflows
- Gate count metrics and optimization quality metrics
- Future: backend-aware native-axis alignment
- Future: drift-aware path selection
- Future: quaternionic error metrics
- Future: Braket circuit support (adapter only, not execution)

---

## What does NOT belong here

- **Canonical quaternion or SU(2) math** — that belongs in `rqm-core`. Do not duplicate it here.
- **Qiskit execution** — that belongs in `rqm-qiskit`.
- **Hardware execution** — never here.
- **Backend calibration mutation** — never here.
- **Full transpilation** — this is an optimizer, not a transpiler.
- **Circuit construction** — that belongs in `rqm-compiler`.

---

## Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `optimizer.py` | Public API, type dispatch, strategy validation, result packaging |
| `fusion.py` | Scanning circuits, identifying single-qubit runs, fusing them |
| `geometry.py` | SU(2) / global-phase normalization helpers (small, rigorous) |
| `metrics.py` | Gate count metrics, matrix error norms, stubs for future quaternionic metrics |
| `qiskit_adapter.py` | Qiskit-specific instruction inspection, matrix extraction, Euler emission |

---

## Public API rules

- The public API lives in `src/rqm_optimize/__init__.py` and is controlled by `__all__`.
- Keep the public API small: `optimize` and `OptimizationResult` are the surface area.
- New public symbols require deliberate addition to `__all__`.
- Do not expose internal helpers in the public API.

---

## Code discipline

- All public functions must have type hints and docstrings.
- All behavior changes require tests.
- No placeholder TODO spam in production code.
- No speculative claims in code comments.
- No dead code.
- Functions should be small and do one thing.
- Prefer readability over cleverness.

---

## Testing

- Tests live in `tests/`.
- Use `pytest` for all tests.
- Equivalence tests must use tolerance-aware comparison (`numpy.allclose` or `Operator` fidelity).
- Circuits with measurements cannot be compared by unitary — compare structure or unitary prefix.
- All regression bugs get a test.

---

## Versioning

- v0.1.x: Qiskit single-qubit run fusion only.
- v0.2.x: Backend-aware native-axis alignment (planned).
- v1.0: Stable public API, Braket support, quaternionic metrics.
