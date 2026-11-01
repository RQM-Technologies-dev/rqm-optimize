# Exact composable fusion: local source candidate

This candidate preserves input circuit scalar phase, accumulates U(2) scalar
phase separately from signed quaternion composition, and carries Qiskit Euler
decomposition phase into the output. Euler simplification is disabled so tiny
rotations are not dropped by a simplification tolerance. A rejected decomposition
retains the original instructions without adding its temporary phase.

The qualified core source is `rqm-core` commit
`6a1c4841528a431a662d2db6aa19503891223078`, based on upstream
`53e7a60b269eb80708afe8a4e28faf45c793531a`. Its exact signed export and tiny-angle
semantics are required; merely finding the same helper names in an older package
is not sufficient. It is an unreleased source candidate with unchanged version
0.3.0. No released core dependency has been selected for this optimizer patch.

## Publication gates

This is not an installable/release-qualified profile. Before merging or releasing:

1. Select a published core version containing the qualified changes and declare
   it in package dependencies. Do not reuse the exploratory `core==0.2.3` pin or
   assume the unchanged candidate version identifies the patch.
2. Reconcile the optimizer's Python >=3.9 declaration/CI matrix with core's
   Python >=3.10 requirement, explicitly documenting any support change.
3. Test a clean installed distribution with the selected dependency, all supported
   Python/Qiskit profiles, and package build/metadata checks.

No version bump, dependency lock, CI/release change or publication is made here.
The existing main-push workflow invokes Release Please; creating a release can
trigger the protected publication workflow. A source qualification is not
permission to create that release.

## Local qualification

With source directories for this candidate and the exact core commit on
`PYTHONPATH`, the existing qualification environment passes 134 tests, including
33 exact-phase regressions. Those tests cover circuit composition, inverse,
controlled embedding and QPY serialization, plus independent Pauli-exponential
checks down to 1e-12 rotations at 2e-15 absolute tolerance. No controlled-core
serializer capability is implied: controlled tests use Qiskit circuit gates.
