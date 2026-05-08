import numpy as np
from kinematics import DHParameters, forward_kinematics


def sample_workspace(
    dh_params: DHParameters,
    n_samples: int = 30_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo workspace sampling.

    Randomly draws joint configurations within joint limits and records
    the end-effector position.

    Returns:
        positions: (n_samples, 3) array of reachable EE positions (mm)
    """
    rng = np.random.default_rng(seed)
    n = len(dh_params.joints)
    low  = np.array([j.theta_min for j in dh_params.joints])
    high = np.array([j.theta_max for j in dh_params.joints])

    positions = np.empty((n_samples, 3))
    for k in range(n_samples):
        q = rng.uniform(low, high)
        _, T_ee = forward_kinematics(dh_params, q)
        positions[k] = T_ee[:3, 3]

    return positions


def workspace_stats(positions: np.ndarray) -> dict:
    """Return min/max reach and bounding box from sampled positions."""
    reach = np.linalg.norm(positions, axis=1)
    return {
        "min_reach": float(reach.min()),
        "max_reach": float(reach.max()),
        "x_range": (float(positions[:, 0].min()), float(positions[:, 0].max())),
        "y_range": (float(positions[:, 1].min()), float(positions[:, 1].max())),
        "z_range": (float(positions[:, 2].min()), float(positions[:, 2].max())),
        "n_samples": len(positions),
    }
