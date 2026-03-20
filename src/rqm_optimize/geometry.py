"""geometry.py — SU(2) and global-phase normalization helpers.

Keeps math helpers small and rigorous.  Does not duplicate rqm-core; this
module only contains the minimum needed to support circuit-level optimization.

Quaternion convention
---------------------
A unit quaternion is represented as a length-4 NumPy array ``q = [w, x, y, z]``
where *w* is the scalar part and *x, y, z* are the pure-imaginary components.

The exact isomorphism between unit quaternions and SU(2) matrices used here is::

    q = (w, x, y, z)  <->  U = [[ w - iz,  -(y + ix)],
                                  [ y - ix,    w + iz ]]

This convention is consistent with the standard quantum rotation gates:

* ``Rx(theta)``  ->  ``(cos(theta/2), sin(theta/2), 0,            0          )``
* ``Ry(theta)``  ->  ``(cos(theta/2), 0,            sin(theta/2), 0          )``
* ``Rz(theta)``  ->  ``(cos(theta/2), 0,            0,            sin(theta/2))``

Quaternion multiplication ``quaternion_multiply(q2, q1)`` represents applying
*q1* first, then *q2*, matching matrix composition ``U2 @ U1``.

For canonicalization the representative with non-negative scalar part (``w >= 0``)
is preferred, keeping the quaternion on the shortest geodesic arc on S³.
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


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def su2_to_quaternion(matrix: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """Return the unit quaternion ``[w, x, y, z]`` for an SU(2) matrix.

    Uses the exact algebraic isomorphism::

        U = [[ w - iz,  -(y + ix)],
             [ y - ix,    w + iz ]]  <->  q = (w, x, y, z)

    The result is normalized and has ``w >= 0`` (canonical shortest-path
    representative on S³).

    Args:
        matrix: 2×2 complex SU(2) array (determinant +1, unitary).

    Returns:
        Length-4 float array ``[w, x, y, z]`` representing the unit quaternion.
    """
    # Extract components from matrix entries:
    #   U[0,0] = w - iz  =>  w = Re(U[0,0]),  z = -Im(U[0,0])
    #   U[0,1] = -(y+ix) =>  y = -Re(U[0,1]), x = -Im(U[0,1])
    w = float(matrix[0, 0].real)
    x = float(-matrix[0, 1].imag)
    y = float(-matrix[0, 1].real)
    z = float(-matrix[0, 0].imag)
    q = np.array([w, x, y, z], dtype=float)
    return quaternion_canonicalize(q)


def quaternion_to_su2(q: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Return the SU(2) matrix for a unit quaternion ``[w, x, y, z]``.

    Inverse of :func:`su2_to_quaternion`.  Uses the isomorphism::

        q = (w, x, y, z)  <->  U = [[ w - iz,  -(y + ix)],
                                      [ y - ix,    w + iz ]]

    Args:
        q: Length-4 float array ``[w, x, y, z]``.

    Returns:
        2×2 complex SU(2) array.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [complex(w, -z), complex(-y, -x)],
            [complex(y, -x), complex(w,  z)],
        ],
        dtype=complex,
    )


def quaternion_multiply(
    q2: NDArray[np.floating],
    q1: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the quaternion product ``q2 * q1``.

    Represents applying gate *q1* first, then gate *q2*, matching matrix
    composition ``U2 @ U1``.

    Uses the standard Hamilton product formula::

        (w2, x2, y2, z2) * (w1, x1, y1, z1) = (
            w2*w1 - x2*x1 - y2*y1 - z2*z1,
            w2*x1 + x2*w1 + y2*z1 - z2*y1,
            w2*y1 - x2*z1 + y2*w1 + z2*x1,
            w2*z1 + x2*y1 - y2*x1 + z2*w1,
        )

    Args:
        q2: Unit quaternion for the second (outer) gate.
        q1: Unit quaternion for the first (inner) gate.

    Returns:
        Product quaternion (not yet canonicalized or re-normalized).
    """
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        dtype=float,
    )


def quaternion_canonicalize(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the canonical unit quaternion with non-negative scalar part.

    Both *q* and *-q* represent the same physical SU(2) rotation.  This
    function normalizes the quaternion and selects the representative with
    ``w >= 0``, keeping it on the shortest geodesic arc (canonical
    representative) on S³.

    Args:
        q: Length-4 float array ``[w, x, y, z]``.

    Returns:
        Normalized quaternion with ``w >= 0``.
    """
    norm = float(np.linalg.norm(q))
    if norm < _ATOL:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_to_axis_angle(
    q: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """Extract the rotation axis and angle from a unit quaternion.

    The physical Bloch-sphere rotation angle is ``theta = 2 * arccos(w)``
    (the factor of 2 arises from the half-angle spinor convention).

    Args:
        q: Length-4 float array ``[w, x, y, z]``, assumed to be a unit
           quaternion.  Canonicalization is applied internally.

    Returns:
        A tuple ``(axis, theta)`` where:

        * ``axis`` is a length-3 float array giving the unit rotation axis
          ``[nx, ny, nz]``.  When the rotation is near-identity (``theta``
          close to 0), ``axis`` defaults to ``[0, 0, 1]``.
        * ``theta`` is the rotation angle in radians (in ``[0, pi]`` after
          canonicalization).
    """
    q = quaternion_canonicalize(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    theta = 2.0 * float(np.arccos(w))
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < _ATOL:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = np.array(q[1:], dtype=float) / sin_half
    return axis, theta


# Reference unit vectors for the three Cartesian rotation axes.
_CARTESIAN_AXES: tuple[tuple[str, NDArray[np.floating]], ...] = (
    ("x", np.array([1.0, 0.0, 0.0])),
    ("y", np.array([0.0, 1.0, 0.0])),
    ("z", np.array([0.0, 0.0, 1.0])),
)

# Default tolerance for axis-alignment checks.
_AXIS_ATOL = 1e-6


def axis_aligned_rotation(
    q: NDArray[np.floating],
    atol: float = _AXIS_ATOL,
) -> tuple[str, float] | None:
    """Return ``(axis_name, theta)`` if *q* is a rotation about x, y, or z.

    Checks whether the rotation axis of *q* lies within *atol* of the x, y, or
    z unit vector.  This is the geometric test that drives the axis-aware
    compression pass: when the result is non-``None``, the caller can emit a
    single named rotation gate (``rx``, ``ry``, or ``rz``) instead of a
    generic ``U`` gate.

    Near-identity rotations (``|theta| < atol``) return ``None`` because the
    axis is numerically undefined at that scale.

    Args:
        q: Length-4 float array ``[w, x, y, z]``, a unit quaternion.
        atol: Absolute tolerance for axis alignment and near-identity detection.

    Returns:
        ``("x"/"y"/"z", theta)`` when the axis is Cartesian-aligned, or
        ``None`` when the rotation is generic or near-identity.
    """
    q = quaternion_canonicalize(q)
    axis, theta = quaternion_to_axis_angle(q)
    if abs(theta) < atol:
        return None
    for name, ref in _CARTESIAN_AXES:
        if np.allclose(axis, ref, atol=atol):
            return name, theta
    return None
