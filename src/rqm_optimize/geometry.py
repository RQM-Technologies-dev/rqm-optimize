"""geometry.py — SU(2) and global-phase normalization helpers.

Keeps math helpers small and rigorous.  Does not duplicate rqm-core; this
module only contains the minimum needed to support circuit-level optimization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_ATOL = 1e-10


def is_unitary(matrix: NDArray[np.complexfloating], atol: float = _ATOL) -> bool:
    """Return True if *matrix* is unitary within *atol*.

    Args:
        matrix: Square complex array.
        atol: Absolute tolerance for the identity comparison.

    Returns:
        True when ``matrix @ matrix†`` is close to the identity.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    eye = np.eye(matrix.shape[0], dtype=complex)
    product = matrix @ matrix.conj().T
    return bool(np.allclose(product, eye, atol=atol))


def remove_global_phase(matrix: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Return a copy of *matrix* with global phase removed (det → +1 for SU(2)).

    For a 2×2 unitary U with det(U) = e^{iφ}, divides by e^{iφ/2} so the
    result has determinant +1 and lives in SU(2).

    Args:
        matrix: 2×2 complex unitary array.

    Returns:
        SU(2)-normalised copy of *matrix*.
    """
    det = np.linalg.det(matrix)
    phase = np.angle(det) / matrix.shape[0]
    return matrix * np.exp(-1j * phase)


def matrices_close(
    a: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    atol: float = _ATOL,
) -> bool:
    """Return True if *a* and *b* are equal up to global phase within *atol*.

    Two unitaries represent the same physical operation when they differ only
    by a global phase.  Uses the Hilbert-Schmidt inner product to check this
    without any angle-wrapping artefacts: if ``a = e^{iα} b`` then
    ``|trace(a† b)| = n`` (the matrix dimension).

    Args:
        a: First unitary matrix (n×n).
        b: Second unitary matrix (n×n).
        atol: Absolute tolerance.

    Returns:
        True when *a* and *b* are equal up to a global phase.
    """
    n = a.shape[0]
    inner = np.trace(a.conj().T @ b)
    return bool(abs(abs(inner) - n) < atol * n)
