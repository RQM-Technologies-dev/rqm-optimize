"""tests/test_geometry_quaternion.py — Tests for quaternion math in geometry.py.

Verifies the exact algebraic isomorphism between unit quaternions and SU(2)
matrices, the quaternion product rule for gate composition, canonicalization,
and axis-angle extraction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rqm_optimize.geometry import (
    matrices_close,
    quaternion_canonicalize,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_su2,
    su2_to_quaternion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rx(theta: float) -> np.ndarray:
    """Standard Rx gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    """Standard Ry gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """Standard Rz gate matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[complex(c, -s), 0], [0, complex(c, s)]], dtype=complex)


def _h() -> np.ndarray:
    """Hadamard gate matrix."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _qnorm(q: np.ndarray) -> float:
    return float(np.linalg.norm(q))


# ---------------------------------------------------------------------------
# su2_to_quaternion / quaternion_to_su2 round-trip
# ---------------------------------------------------------------------------

def test_identity_roundtrip() -> None:
    """Identity matrix maps to identity quaternion and back."""
    eye = np.eye(2, dtype=complex)
    q = su2_to_quaternion(eye)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)
    recovered = quaternion_to_su2(q)
    assert np.allclose(recovered, eye, atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi])
def test_rx_quaternion_form(theta: float) -> None:
    """Rx(theta) maps to quaternion (cos(theta/2), sin(theta/2), 0, 0)."""
    mat = _rx(theta)
    q = su2_to_quaternion(mat)
    # Must be unit quaternion.
    assert abs(_qnorm(q) - 1.0) < 1e-10
    # Either q or -q should match the expected form.
    expected = np.array([math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0])
    # After canonicalization (w >= 0) both q and expected share the same sign.
    expected = quaternion_canonicalize(expected)
    assert np.allclose(q, expected, atol=1e-10), f"theta={theta}: q={q} expected={expected}"


@pytest.mark.parametrize("theta", [0.0, math.pi / 3, math.pi / 2, math.pi])
def test_ry_quaternion_form(theta: float) -> None:
    """Ry(theta) maps to quaternion (cos(theta/2), 0, sin(theta/2), 0)."""
    mat = _ry(theta)
    q = su2_to_quaternion(mat)
    expected = quaternion_canonicalize(
        np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])
    )
    assert np.allclose(q, expected, atol=1e-10)


@pytest.mark.parametrize("theta", [0.0, math.pi / 6, math.pi / 2, math.pi])
def test_rz_quaternion_form(theta: float) -> None:
    """Rz(theta) maps to quaternion (cos(theta/2), 0, 0, sin(theta/2))."""
    mat = _rz(theta)
    q = su2_to_quaternion(mat)
    expected = quaternion_canonicalize(
        np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])
    )
    assert np.allclose(q, expected, atol=1e-10)


def test_hadamard_quaternion_form() -> None:
    """H is a pi-rotation about the (x+z)/sqrt(2) axis.

    Expected quaternion: (0, 1/sqrt(2), 0, 1/sqrt(2)) up to sign/canonicalization.
    Since H has det=-1 (not SU(2)), global phase must be removed first.
    """
    from rqm_optimize.geometry import remove_global_phase

    mat = _h()
    mat_su2 = remove_global_phase(mat)
    q = su2_to_quaternion(mat_su2)
    # Hadamard: theta = pi, axis = (1/sqrt2, 0, 1/sqrt2).
    # q = (cos(pi/2), sin(pi/2)/sqrt2, 0, sin(pi/2)/sqrt2) = (0, 1/sqrt2, 0, 1/sqrt2)
    # w=0 so sign ambiguity exists; verify via matrix reconstruction.
    recovered = quaternion_to_su2(q)
    assert matrices_close(mat_su2, recovered, atol=1e-10)


def test_roundtrip_arbitrary_su2() -> None:
    """A random SU(2) matrix round-trips through su2_to_quaternion/quaternion_to_su2."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Generate a random unitary with det=+1.
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        lam = rng.uniform(0, 2 * math.pi)
        # Use standard U-gate form.
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        mat = np.array(
            [
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
            ],
            dtype=complex,
        )
        # Normalize to SU(2) (det → +1).
        det = np.linalg.det(mat)
        mat /= np.sqrt(det)

        q = su2_to_quaternion(mat)
        recovered = quaternion_to_su2(q)
        # Must match up to global phase.
        assert matrices_close(mat, recovered, atol=1e-9), "Round-trip failed for random SU(2)"


# ---------------------------------------------------------------------------
# quaternion_canonicalize
# ---------------------------------------------------------------------------

def test_canonicalize_positive_w() -> None:
    """A quaternion with w>0 is returned unchanged (after normalization)."""
    q = np.array([0.6, 0.2, 0.5, 0.3])
    q = q / np.linalg.norm(q)
    qc = quaternion_canonicalize(q)
    assert qc[0] >= 0.0
    assert np.allclose(qc, q)


def test_canonicalize_negative_w_flipped() -> None:
    """A quaternion with w<0 is negated so that w>=0."""
    q = np.array([-0.6, 0.2, 0.5, 0.3])
    q = q / np.linalg.norm(q)
    qc = quaternion_canonicalize(q)
    assert qc[0] >= 0.0
    assert np.allclose(qc, -q / np.linalg.norm(q))


def test_canonicalize_normalizes() -> None:
    """quaternion_canonicalize normalizes an unnormalized quaternion."""
    q = np.array([3.0, 0.0, 0.0, 0.0])
    qc = quaternion_canonicalize(q)
    assert abs(_qnorm(qc) - 1.0) < 1e-10
    assert np.allclose(qc, [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# quaternion_multiply — gate composition
# ---------------------------------------------------------------------------

def test_multiply_identity_left() -> None:
    """Identity * q == q."""
    q = su2_to_quaternion(_rx(0.5))
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = quaternion_multiply(identity, q)
    assert np.allclose(result, q, atol=1e-12)


def test_multiply_identity_right() -> None:
    """q * identity == q."""
    q = su2_to_quaternion(_ry(0.7))
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = quaternion_multiply(q, identity)
    assert np.allclose(result, q, atol=1e-12)


def test_multiply_matches_matrix_composition() -> None:
    """quaternion_multiply(q2, q1) produces the same SU(2) as U2 @ U1."""
    from rqm_optimize.geometry import remove_global_phase

    angles = [(0.5, 0.3), (math.pi / 3, math.pi / 7), (1.2, 0.9)]
    for theta1, theta2 in angles:
        u1 = _rx(theta1)
        u2 = _ry(theta2)
        q1 = su2_to_quaternion(remove_global_phase(u1))
        q2 = su2_to_quaternion(remove_global_phase(u2))
        q_prod = quaternion_canonicalize(quaternion_multiply(q2, q1))
        u_prod = quaternion_to_su2(q_prod)
        u_direct = remove_global_phase(u2 @ u1)
        assert matrices_close(u_prod, u_direct, atol=1e-9), (
            f"Matrix mismatch for theta1={theta1}, theta2={theta2}"
        )


def test_multiply_three_gates_matches_matrix() -> None:
    """Three-gate quaternion chain matches three-matrix product."""
    from rqm_optimize.geometry import remove_global_phase

    u1, u2, u3 = _rx(0.4), _ry(0.6), _rz(0.2)
    q1 = su2_to_quaternion(remove_global_phase(u1))
    q2 = su2_to_quaternion(remove_global_phase(u2))
    q3 = su2_to_quaternion(remove_global_phase(u3))

    q_total = quaternion_canonicalize(quaternion_multiply(q3, quaternion_multiply(q2, q1)))
    u_quat = quaternion_to_su2(q_total)
    u_matrix = remove_global_phase(u3 @ u2 @ u1)
    assert matrices_close(u_quat, u_matrix, atol=1e-9)


def test_multiply_noncommutativity() -> None:
    """Rx then Ry is different from Ry then Rx (non-commutative)."""
    from rqm_optimize.geometry import remove_global_phase

    theta = math.pi / 4
    q_x = su2_to_quaternion(_rx(theta))
    q_y = su2_to_quaternion(_ry(theta))

    q_xy = quaternion_multiply(q_y, q_x)  # Rx first, Ry second
    q_yx = quaternion_multiply(q_x, q_y)  # Ry first, Rx second

    # Products must differ.
    assert not np.allclose(q_xy, q_yx, atol=1e-6)
    # But their matrices must individually match the direct matrix products.
    u_xy = quaternion_to_su2(quaternion_canonicalize(q_xy))
    u_yx = quaternion_to_su2(quaternion_canonicalize(q_yx))
    assert matrices_close(u_xy, remove_global_phase(_ry(theta) @ _rx(theta)), atol=1e-9)
    assert matrices_close(u_yx, remove_global_phase(_rx(theta) @ _ry(theta)), atol=1e-9)


def test_multiply_same_axis_adds_angles() -> None:
    """Rx(a) followed by Rx(b) == Rx(a+b) up to global phase."""
    a, b = 0.5, 0.7
    q_a = su2_to_quaternion(_rx(a))
    q_b = su2_to_quaternion(_rx(b))
    q_sum = quaternion_canonicalize(quaternion_multiply(q_b, q_a))
    q_direct = su2_to_quaternion(_rx(a + b))
    assert np.allclose(q_sum, q_direct, atol=1e-9)


# ---------------------------------------------------------------------------
# quaternion_to_axis_angle
# ---------------------------------------------------------------------------

def test_axis_angle_identity() -> None:
    """Identity quaternion gives theta=0 and a default axis."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    axis, theta = quaternion_to_axis_angle(q)
    assert abs(theta) < 1e-10
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-10


@pytest.mark.parametrize(
    "theta, expected_axis",
    [
        (math.pi / 2, np.array([1.0, 0.0, 0.0])),
        (math.pi / 3, np.array([1.0, 0.0, 0.0])),
        (math.pi,     np.array([1.0, 0.0, 0.0])),
    ],
)
def test_axis_angle_rx(theta: float, expected_axis: np.ndarray) -> None:
    """Rx(theta) has rotation axis x̂ and rotation angle theta."""
    mat = _rx(theta)
    q = su2_to_quaternion(mat)
    axis, angle = quaternion_to_axis_angle(q)
    assert abs(angle - theta) < 1e-9, f"angle mismatch: {angle} vs {theta}"
    assert np.allclose(axis, expected_axis, atol=1e-9), f"axis mismatch: {axis}"


@pytest.mark.parametrize("theta", [math.pi / 4, math.pi / 2, math.pi])
def test_axis_angle_ry(theta: float) -> None:
    """Ry(theta) has rotation axis ŷ and rotation angle theta."""
    mat = _ry(theta)
    q = su2_to_quaternion(mat)
    axis, angle = quaternion_to_axis_angle(q)
    assert abs(angle - theta) < 1e-9
    assert np.allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 2, math.pi])
def test_axis_angle_rz(theta: float) -> None:
    """Rz(theta) has rotation axis ẑ and rotation angle theta."""
    mat = _rz(theta)
    q = su2_to_quaternion(mat)
    axis, angle = quaternion_to_axis_angle(q)
    assert abs(angle - theta) < 1e-9
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)


def test_axis_angle_reconstruction() -> None:
    """axis_angle → quaternion round-trip via axis-angle-to-quaternion formula."""
    for theta in [0.3, 0.7, math.pi / 2, math.pi]:
        for axis_dir in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / math.sqrt(2),
        ]:
            # Build quaternion from axis-angle.
            q_orig = np.array(
                [math.cos(theta / 2), *(axis_dir * math.sin(theta / 2))],
                dtype=float,
            )
            q_orig = quaternion_canonicalize(q_orig)
            # Extract axis-angle back.
            axis_out, theta_out = quaternion_to_axis_angle(q_orig)
            # Rebuild quaternion.
            q_rebuild = np.array(
                [math.cos(theta_out / 2), *(axis_out * math.sin(theta_out / 2))],
                dtype=float,
            )
            q_rebuild = quaternion_canonicalize(q_rebuild)
            assert np.allclose(q_orig, q_rebuild, atol=1e-9), (
                f"Round-trip failed: theta={theta}, axis={axis_dir}"
            )
