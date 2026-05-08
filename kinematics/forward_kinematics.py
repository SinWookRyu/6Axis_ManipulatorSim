import numpy as np
from .dh_params import DHParameters


def dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """
    Standard DH transformation matrix.
    T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st * ca,  st * sa,  a * ct],
        [st,     ct * ca, -ct * sa,  a * st],
        [0.0,    sa,       ca,       d     ],
        [0.0,    0.0,      0.0,      1.0   ],
    ])


def forward_kinematics(dh_params: DHParameters, joint_angles) -> tuple:
    """
    Compute forward kinematics for all joints.

    Returns:
        T_list: list of 4x4 matrices [T_base, T_0_1, T_0_2, ..., T_0_6]
        T_ee:   4x4 end-effector transform (same as T_list[-1])
    """
    q = np.asarray(joint_angles, dtype=float)
    T = np.eye(4)
    T_list = [T.copy()]
    for i, joint in enumerate(dh_params.joints):
        theta = q[i] + joint.theta_offset
        T = T @ dh_matrix(joint.a, joint.d, joint.alpha, theta)
        T_list.append(T.copy())
    return T_list, T_list[-1]


def get_joint_positions(dh_params: DHParameters, joint_angles) -> np.ndarray:
    """Return (N+1, 3) array of joint origin positions in world frame."""
    T_list, _ = forward_kinematics(dh_params, joint_angles)
    return np.array([T[:3, 3] for T in T_list])


def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → Roll-Pitch-Yaw (ZYX convention), radians."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if abs(np.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-Pitch-Yaw (ZYX convention) → rotation matrix."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp],  [0, 1, 0],  [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],    [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def make_transform(position, rpy=None, R=None) -> np.ndarray:
    """Build a 4x4 transform from position + RPY or rotation matrix."""
    T = np.eye(4)
    T[:3, 3] = position
    if R is not None:
        T[:3, :3] = R
    elif rpy is not None:
        T[:3, :3] = rpy_to_rotation_matrix(*rpy)
    return T
