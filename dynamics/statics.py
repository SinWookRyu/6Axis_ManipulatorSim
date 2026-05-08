import numpy as np
from kinematics import geometric_jacobian, DHParameters

# Approximate rated torques for FANUC M-10iA joints (Nm)
FANUC_RATED_TORQUES = np.array([588.0, 588.0, 294.0, 117.0, 117.0, 58.0])


def compute_joint_torques(
    dh_params: DHParameters,
    joint_angles,
    external_wrench,
) -> np.ndarray:
    """
    Compute joint torques due to an external wrench at the end-effector.

    Principle of virtual work:  τ = J^T · F_ext

    Args:
        external_wrench: [Fx, Fy, Fz, Mx, My, Mz] (N, N·m)

    Returns:
        torques: joint torques (N·m), shape (n,)
    """
    J = geometric_jacobian(dh_params, joint_angles)
    return J.T @ np.asarray(external_wrench, dtype=float)


def joint_load_report(
    dh_params: DHParameters,
    joint_angles,
    external_wrench,
    rated_torques=None,
) -> tuple:
    """
    Full joint load analysis.

    Returns:
        torques:     np.ndarray (n,)  — computed torques (N·m)
        report:      list of dicts per joint
    """
    if rated_torques is None:
        n = len(dh_params.joints)
        rated_torques = (
            FANUC_RATED_TORQUES[:n]
            if n <= len(FANUC_RATED_TORQUES)
            else np.full(n, 100.0)
        )

    torques = compute_joint_torques(dh_params, joint_angles, external_wrench)
    report = []
    for i, (tau, rated) in enumerate(zip(torques, rated_torques)):
        report.append({
            "joint": i + 1,
            "torque_nm": float(tau),
            "rated_nm": float(rated),
            "load_pct": abs(float(tau)) / float(rated) * 100.0,
        })
    return torques, report
