# Changelog

All notable changes to `rqm-optimize` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.4](https://github.com/RQM-Technologies-dev/rqm-optimize/compare/v0.1.3...v0.1.4) (2026-07-29)


### Documentation

* point optimizer formats to the canonical API ([#13](https://github.com/RQM-Technologies-dev/rqm-optimize/issues/13)) ([09a17b5](https://github.com/RQM-Technologies-dev/rqm-optimize/commit/09a17b5c045e6a368d74a0a3af17165e48e66a0b))

## [0.1.3](https://github.com/RQM-Technologies-dev/rqm-optimize/compare/v0.1.2...v0.1.3) (2026-07-29)


### Bug Fixes

* dispatch protected publication by repository ([#12](https://github.com/RQM-Technologies-dev/rqm-optimize/issues/12)) ([3e63eae](https://github.com/RQM-Technologies-dev/rqm-optimize/commit/3e63eae897a1115ff030bbd536e05724fb31a73a))
* dispatch release pull request CI reliably ([#8](https://github.com/RQM-Technologies-dev/rqm-optimize/issues/8)) ([6d3fbe1](https://github.com/RQM-Technologies-dev/rqm-optimize/commit/6d3fbe17a4cc676acdd8f42811f37b845c7c26d7))
* keep generated releases verifiable ([#10](https://github.com/RQM-Technologies-dev/rqm-optimize/issues/10)) ([e85eaa0](https://github.com/RQM-Technologies-dev/rqm-optimize/commit/e85eaa0358069d3b03ba82afae0a6aedfbf021ae))


### Documentation

* correct quaternion information claims ([#11](https://github.com/RQM-Technologies-dev/rqm-optimize/issues/11)) ([c8e0c26](https://github.com/RQM-Technologies-dev/rqm-optimize/commit/c8e0c26f579e68446f9949114c8ac30229471d42))

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
