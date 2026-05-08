import numpy as np
from .dh_params import DHParameters
from .forward_kinematics import forward_kinematics


def geometric_jacobian(dh_params: DHParameters, joint_angles) -> np.ndarray:
    """
    Compute the 6×N geometric Jacobian for revolute joints.

    For revolute joint i:
        Jv_i = z_{i-1} × (p_e - p_{i-1})   (linear velocity)
        Jw_i = z_{i-1}                        (angular velocity)
    """
    n = len(dh_params.joints)
    q = np.asarray(joint_angles, dtype=float)
    T_list, T_ee = forward_kinematics(dh_params, q)
    p_ee = T_ee[:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        z = T_list[i][:3, 2]
        p = T_list[i][:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J
