# Exact composable fusion qualification

This change preserves input circuit scalar phase, accumulates U(2) scalar phase
separately from signed quaternion composition, and carries Qiskit Euler
phase into the output. Euler simplification is disabled so tiny rotations are
not dropped by its simplification tolerance. A rejected decomposition retains
the original instructions without adding its temporary phase.

## Published dependency and compatibility

The required implementation is published in `rqm-core` 0.3.1; this package
requires `rqm-core>=0.3.1,<0.4`. The qualified public PyPI wheel SHA256 is
`443f3f702fad31989b054fa7734cb0cdcff807b251e77ab8a4565a64d91ab48b`.
It contains the signed-export and tiny-angle behavior first qualified at core
source commit `6a1c4841528a431a662d2db6aa19503891223078`.

Python support is now >=3.10, matching core's minimum. Python 3.9 is removed
from the declared classifiers and CI matrix. This is an explicit compatibility
change, not a skipped test. Optimizer version remains 0.1.3 on the feature branch;
Release Please owns the subsequent release version.

## Validation

The source profile passed 134 tests. A separate Python 3.12 environment also
passes all 134 tests using built optimizer and published core wheels installed
into site-packages, without PYTHONPATH/source overlays. This includes 33 exact
phase regressions covering composition, inverse, controlled embedding and QPY
serialization, plus independent Pauli-exponential checks down to 1e-12 rotations
at 2e-15 absolute tolerance. Controlled tests use Qiskit circuit gates; they do
not qualify a controlled core serializer. On unchanged upstream, 32 of these
33 new checks failed, demonstrating the repaired defects.

The isolated installed profile passes pip dependency consistency checking,
sdist/wheel build and twine metadata checks. Linux Python 3.10–3.12 CI must still
pass on the final integrated commit. The earlier draft CI failed because core
was undeclared; Python 3.9 separately failed runner setup before tests. Those
historical failures are preserved in the external qualification evidence.

No publication is performed by local qualification. Main pushes invoke Release
Please; its release PR must include the final dependency fix before release.
