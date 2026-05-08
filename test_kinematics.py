import sys
sys.path.insert(0, ".")

from kinematics import (
    DHParameters, forward_kinematics, geometric_jacobian,
    inverse_kinematics_dls, SingularityDetector,
)
from dynamics import joint_load_report
from workspace import sample_workspace, workspace_stats
import numpy as np

dh = DHParameters.fanuc_m10ia()
print(f"DH params: {dh.name}, {len(dh)} joints")

q = np.zeros(6)
T_list, T_ee = forward_kinematics(dh, q)
print(f"FK home EE pos: {T_ee[:3,3].round(2)} mm")

J = geometric_jacobian(dh, q)
print(f"Jacobian shape: {J.shape}")
assert J.shape == (6, 6)

sd = SingularityDetector()
res = sd.check(dh, q)
print(f"Singularity: {res['label']}")

T_target = T_ee.copy()
T_target[0, 3] += 50
q_ik, ok, err = inverse_kinematics_dls(dh, T_target, q_init=q)
print(f"IK test: converged={ok}, error={err:.3f} mm")

torques, report = joint_load_report(dh, q, [10, 0, 0, 0, 0, 0])
print(f"Torques (J1-J3): {[round(r['torque_nm'],2) for r in report[:3]]}")

pts = sample_workspace(dh, n_samples=500)
stats = workspace_stats(pts)
print(f"Workspace: max_reach={stats['max_reach']:.0f} mm, N={stats['n_samples']}")

print("\nALL TESTS PASSED")
