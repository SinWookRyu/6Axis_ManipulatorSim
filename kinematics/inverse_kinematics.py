import numpy as np
from .dh_params import DHParameters
from .forward_kinematics import forward_kinematics
from .jacobian import geometric_jacobian


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Compute 6D pose error [dp, do] between current and target transforms.
    Orientation error uses the cross-product method (stable near current pose).
    """
    dp = T_target[:3, 3] - T_current[:3, 3]
    Rc, Rd = T_current[:3, :3], T_target[:3, :3]
    do = 0.5 * (
        np.cross(Rc[:, 0], Rd[:, 0]) +
        np.cross(Rc[:, 1], Rd[:, 1]) +
        np.cross(Rc[:, 2], Rd[:, 2])
    )
    return np.concatenate([dp, do])


def inverse_kinematics_dls(
    dh_params: DHParameters,
    target_T: np.ndarray,
    q_init=None,
    max_iter: int = 150,
    tol: float = 1.0,
    lambda_sq: float = 0.01,
    alpha: float = 0.5,
) -> tuple:
    """
    Damped Least Squares (Levenberg–Marquardt) IK solver.

    dq = J^T (J J^T + λ²I)^{-1} δx

    Args:
        tol:       convergence threshold (position error in mm)
        lambda_sq: fixed damping factor (use adaptive_dls for variable damping)
        alpha:     step size scale

    Returns:
        (q, success, final_error_norm)
    """
    n = len(dh_params.joints)
    q = np.zeros(n) if q_init is None else np.array(q_init, dtype=float)

    for _ in range(max_iter):
        _, T_curr = forward_kinematics(dh_params, q)
        err = pose_error(T_curr, target_T)
        err_norm = np.linalg.norm(err[:3])

        if err_norm < tol:
            return q, True, err_norm

        J = geometric_jacobian(dh_params, q)

        # Adaptive damping: increase near singularity
        svs = np.linalg.svd(J, compute_uv=False)
        w_min = svs[-1]
        eps = 0.05
        lam2 = lambda_sq if w_min > eps else lambda_sq + (1 - w_min / eps) ** 2 * 0.5

        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lam2 * np.eye(6), err)

        # Clamp step size
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.15:
            dq *= 0.15 / dq_norm

        q = q + alpha * dq

        # Apply joint limits
        for i, joint in enumerate(dh_params.joints):
            q[i] = np.clip(q[i], joint.theta_min, joint.theta_max)

    _, T_curr = forward_kinematics(dh_params, q)
    err = pose_error(T_curr, target_T)
    return q, False, np.linalg.norm(err[:3])


def ik_fast(
    dh_params: DHParameters,
    target_T: np.ndarray,
    q_init=None,
) -> tuple:
    """Fast IK for real-time control: fewer iterations, larger tolerance."""
    return inverse_kinematics_dls(
        dh_params, target_T, q_init=q_init,
        max_iter=20, tol=2.0, lambda_sq=0.01, alpha=0.8,
    )
