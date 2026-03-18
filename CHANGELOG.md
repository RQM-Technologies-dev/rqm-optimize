# Changelog

All notable changes to `rqm-optimize` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2026-03-18

### Added

- `optimize(circuit, ...)` — public entry point for single-qubit run fusion.
- `OptimizationResult` — dataclass returned when `return_metadata=True`.
- Single-qubit run detection and matrix fusion via `fusion.py`.
- SU(2) / global-phase normalisation helpers in `geometry.py`.
- Gate count, circuit depth, and single-qubit gate count metrics in `metrics.py`.
- Qiskit instruction inspection, matrix extraction, and Euler gate emission in `qiskit_adapter.py`.
- `native_basis` parameter: supports `None` (compact `u` gate), `"ibm"` (`rz`/`sx`), and `"zyz"` (`rz`/`ry`).
- `backend` argument: auto-infers IBM native basis from `BackendV2.operation_names` or `BackendV1.configuration().basis_gates`.
- `py.typed` PEP 561 marker for typed distributions.
- `__version__` attribute populated from installed package metadata.
- Full test suite covering fusion correctness, boundary preservation, metadata fields, native-basis paths, backend inference, and determinism.

### Supported gates in v0.1

`rx`, `ry`, `rz`, `u`, `u3`, `u2`, `u1`, `p`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `id`, `sx`, `sxdg`, `r`, and any single-qubit `UnitaryGate` whose matrix can be extracted.

[0.1.0]: https://github.com/RQM-Technologies-dev/rqm-optimize/releases/tag/v0.1.0
