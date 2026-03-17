"""tests/test_public_api.py — Verify importability and __all__ contract."""

import rqm_optimize


def test_optimize_importable() -> None:
    from rqm_optimize import optimize  # noqa: F401

    assert callable(optimize)


def test_optimization_result_importable() -> None:
    from rqm_optimize import OptimizationResult  # noqa: F401

    assert OptimizationResult is not None


def test_all_contains_expected_symbols() -> None:
    assert "optimize" in rqm_optimize.__all__
    assert "OptimizationResult" in rqm_optimize.__all__


def test_all_has_no_extra_symbols() -> None:
    assert set(rqm_optimize.__all__) == {"optimize", "OptimizationResult"}


def test_optimize_raises_on_wrong_type() -> None:
    import pytest

    from rqm_optimize import optimize

    with pytest.raises(TypeError, match="QuantumCircuit"):
        optimize("not a circuit")  # type: ignore[arg-type]


def test_optimize_raises_on_unknown_strategy() -> None:
    import pytest
    from qiskit import QuantumCircuit

    from rqm_optimize import optimize

    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="strategy"):
        optimize(qc, strategy="unknown_strategy")
