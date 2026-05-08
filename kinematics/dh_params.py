import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class JointDH:
    """Standard DH parameters for a single revolute joint."""
    a: float            # Link length (mm)
    d: float            # Link offset (mm)
    alpha: float        # Twist angle (rad)
    theta_offset: float # Joint zero offset (rad)
    theta_min: float    # Minimum joint angle (rad)
    theta_max: float    # Maximum joint angle (rad)
    name: str = ""


@dataclass
class DHParameters:
    """DH parameter set for a 6-DOF serial manipulator."""
    joints: List[JointDH]
    name: str = "Robot"

    def __len__(self):
        return len(self.joints)

    def __iter__(self):
        return iter(self.joints)

    def copy(self):
        new_joints = [JointDH(**vars(j)) for j in self.joints]
        return DHParameters(joints=new_joints, name=self.name)

    @classmethod
    def fanuc_m10ia(cls):
        """FANUC M-10iA approximate DH parameters (Standard DH convention).

        T_i = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
        Actual joint angle applied: theta_dh = joint_angle + theta_offset
        """
        deg = np.radians
        joints = [
            JointDH(a=150,  d=520, alpha=deg(-90), theta_offset=0,
                    theta_min=deg(-170), theta_max=deg(170),  name="J1"),
            JointDH(a=640,  d=0,   alpha=0,         theta_offset=deg(-90),
                    theta_min=deg(-100), theta_max=deg(135),  name="J2"),
            JointDH(a=200,  d=0,   alpha=deg(-90), theta_offset=deg(90),
                    theta_min=deg(-120), theta_max=deg(270),  name="J3"),
            JointDH(a=0,    d=700, alpha=deg(90),  theta_offset=0,
                    theta_min=deg(-190), theta_max=deg(190),  name="J4"),
            JointDH(a=0,    d=0,   alpha=deg(-90), theta_offset=0,
                    theta_min=deg(-120), theta_max=deg(120),  name="J5"),
            JointDH(a=0,    d=115, alpha=0,         theta_offset=0,
                    theta_min=deg(-360), theta_max=deg(360),  name="J6"),
        ]
        return cls(joints=joints, name="FANUC M-10iA")
