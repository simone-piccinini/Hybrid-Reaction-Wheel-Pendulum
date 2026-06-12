"""Validation of the plant linearisation against reference linear algebra.

model.md's "Practical cautions" prescribe two sanity checks that need
eigenvalues and matrix rank — banned in src/, allowed here: the open-loop A
must be unstable (the pendulum wants to fall), and (A, B) must be controllable
except for the structurally uncontrollable wheel angle. Also cross-checks the
plant's ZOH discretisation against scipy.signal.cont2discrete.
"""

import numpy as np
from scipy.signal import cont2discrete

from inverted_pendulum.numerics.constants import ATOL, VALIDATION_RTOL
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel


def make_plant() -> ReactionWheelPendulum:
    return ReactionWheelPendulum(
        pendulum_mass=0.3,
        pendulum_length=0.15,
        body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def test_open_loop_is_unstable():
    # model.md: "The open-loop A should be unstable ... A fully stable A means
    # a sign or assembly error."
    A = make_plant().linearize().A
    assert np.max(np.linalg.eigvals(A).real) > 0.0


def test_angle_and_rate_states_are_reachable():
    # model.md's binding requirement: "the angle/rate states must be reachable
    # through the motor" — the reduced [theta_p, theta_p_dot, theta_w_dot]
    # system must be fully controllable.
    model = make_plant().linearize()
    keep = [0, 1, 3]
    A_r = model.A[np.ix_(keep, keep)]
    B_r = model.B[keep, :]
    assert np.linalg.matrix_rank(controllability_matrix(A_r, B_r)) == 3


def test_full_state_controllability_rank():
    # model.md remarks the wheel angle is uncontrollable, but that holds only
    # for momentum-conserving plants (no gravity coupling). Here gravity is an
    # external torque and theta_w is reached through the theta_w_dot
    # integrator chain, so the Kalman rank is genuinely 4 (the smallest
    # singular value of the controllability matrix is O(1), not round-off).
    # model.md treats uncontrollability as acceptable ("is fine"), not
    # required, so full rank satisfies its caution a fortiori.
    model = make_plant().linearize()
    ctrb = controllability_matrix(model.A, model.B)
    assert np.linalg.matrix_rank(ctrb) == 4
    assert np.linalg.svd(ctrb, compute_uv=False)[-1] > 1.0


def test_zoh_discretization_matches_scipy():
    # model.md "Implementation order" step 5: discretise (A, B) at the control
    # sampling period via the matrix exponential.
    model = make_plant().linearize()
    dt = 0.01
    discrete = model.discretize(dt)
    A_d_ref, B_d_ref, *_ = cont2discrete(
        (model.A, model.B, model.C, model.D), dt, method="zoh"
    )
    assert np.allclose(discrete.A, A_d_ref, rtol=VALIDATION_RTOL, atol=ATOL)
    assert np.allclose(discrete.B, B_d_ref, rtol=VALIDATION_RTOL, atol=ATOL)
