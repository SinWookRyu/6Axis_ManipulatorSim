import numpy as np
from .dh_params import DHParameters
from .jacobian import geometric_jacobian


class SingularityDetector:
    """Detects singularity configurations and classifies them."""

    # Thresholds
    SINGULAR_SV_THRESH = 0.5     # min singular value → singular
    NEAR_SINGULAR_SV = 5.0       # min singular value → near-singular warning
    SINGULAR_COND = 200.0        # condition number → singular
    NEAR_SINGULAR_COND = 50.0    # condition number → warning

    def singular_values(self, J: np.ndarray) -> np.ndarray:
        return np.linalg.svd(J, compute_uv=False)

    def manipulability(self, J: np.ndarray) -> float:
        """Yoshikawa manipulability: w = sqrt(det(J J^T))."""
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def condition_number(self, J: np.ndarray) -> float:
        svs = self.singular_values(J)
        return float(svs[0] / svs[-1]) if svs[-1] > 1e-12 else np.inf

    def check(self, dh_params: DHParameters, joint_angles) -> dict:
        """
        Full singularity report.

        Returns dict:
            is_singular  : bool
            near_singular: bool
            manip        : float  (manipulability)
            cond         : float  (condition number)
            min_sv       : float  (minimum singular value)
            label        : str    (human-readable status)
            color        : str    ('green' / 'orange' / 'red')
            type         : str    (wrist / elbow / shoulder / none)
        """
        q = np.asarray(joint_angles, dtype=float)
        J = geometric_jacobian(dh_params, q)
        svs = self.singular_values(J)
        w = self.manipulability(J)
        cond = self.condition_number(J)
        min_sv = float(svs[-1])

        is_sing = min_sv < self.SINGULAR_SV_THRESH or cond > self.SINGULAR_COND
        near = not is_sing and (
            min_sv < self.NEAR_SINGULAR_SV or cond > self.NEAR_SINGULAR_COND
        )

        sing_type = self._classify(q)

        if is_sing:
            label = f"[SINGULAR] {sing_type} | sv={min_sv:.3f} | cond={cond:.0f}"
            color = "red"
        elif near:
            label = f"[Warning] near {sing_type} | sv={min_sv:.3f} | cond={cond:.0f}"
            color = "orange"
        else:
            label = f"OK | manip={w:.1f} | sv={min_sv:.2f}"
            color = "green"

        return dict(
            is_singular=is_sing, near_singular=near,
            manip=w, cond=cond, min_sv=min_sv,
            label=label, color=color, type=sing_type,
        )

    def _classify(self, q) -> str:
        """Heuristic singularity type classification."""
        j5 = q[4]  # J5 angle
        j3 = q[2]  # J3 angle
        if abs(j5) < np.radians(5):
            return "wrist"
        if abs(j3) < np.radians(5) or abs(j3 - np.pi) < np.radians(5):
            return "elbow"
        return "none"

    def avoidance_dls(self, J: np.ndarray, dx: np.ndarray,
                      eps: float = 0.05, lam_max_sq: float = 0.5) -> np.ndarray:
        """
        Variable Damping (Nakamura & Hanafusa) for singularity avoidance.
        λ²(w) = lam_max_sq * (1 - w/eps)²  when w < eps, else 0
        """
        svs = self.singular_values(J)
        w = float(svs[-1])
        lam_sq = lam_max_sq * (1 - w / eps) ** 2 if w < eps else 0.0
        JJT = J @ J.T
        return J.T @ np.linalg.solve(JJT + lam_sq * np.eye(6), dx)
