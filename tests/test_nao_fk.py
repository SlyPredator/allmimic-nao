"""Gatekeeper test: Forward Kinematics parity between PyTorch batched FK, MuJoCo, and NAOKinematics.

Evaluates:
  1. TreeKinematics vs MuJoCo mj_kinematics across 1,000 random joint configurations.
  2. FastTreeKinematics vs TreeKinematics parity (supporting non-principal joint axes).
  3. Site tracking accuracy for all 9 SuperMimic tracking sites (< 10^-5 m position error).
  4. Parity with third_party/nao_kinematics (analytical DH oracle) in Torso frame.
"""

import sys
from pathlib import Path
import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "nao_kinematics")]

from allmimic.kinematics import TreeKinematics, FastTreeKinematics
from allmimic.rotations import quat_to_mat

MODEL_PATH = ROOT / "assets" / "nao" / "nao_supermimic.xml"
TRACKING_SITES = [
    "torso_anchor",
    "head_track",
    "l_hand_track",
    "r_hand_track",
    "l_elbow_track",
    "r_elbow_track",
    "l_knee_track",
    "r_knee_track",
    "l_ankle_track",
    "r_ankle_track",
]


@pytest.fixture(scope="module")
def mj_model():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_nao_fk_gatekeeper_1000_poses(mj_model):
    """Hard Gate: Max L-infinity error < 1e-5 m (pos) and < 1e-4 rad (quat)."""
    m = mj_model
    d = mujoco.MjData(m)
    tk = TreeKinematics(m, "cpu")
    ftk = FastTreeKinematics(m, "cpu")

    site_indices = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in TRACKING_SITES]
    hyp_l = m.joint("LHipYawPitch").qposadr - 7
    hyp_r = m.joint("RHipYawPitch").qposadr - 7

    np.random.seed(42)
    max_body_pos_err = 0.0
    max_body_rot_err = 0.0
    max_site_pos_err = 0.0
    max_site_rot_err = 0.0
    max_fast_diff = 0.0

    num_poses = 1000
    for _ in range(num_poses):
        root_pos = np.random.uniform(-1.0, 1.0, size=3)
        root_quat = np.random.uniform(-1.0, 1.0, size=4)
        root_quat /= np.linalg.norm(root_quat)
        q = np.random.uniform(-1.2, 1.2, size=24)
        # Enforce pelvis equality
        hyp_val = np.random.uniform(-0.8, 0.5)
        q[hyp_l] = hyp_val
        q[hyp_r] = hyp_val

        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        d.qpos[7:] = q
        mujoco.mj_kinematics(m, d)

        t_root_pos = torch.tensor(root_pos, dtype=torch.float32)
        t_root_rot = quat_to_mat(torch.tensor(root_quat, dtype=torch.float32))
        t_q = torch.tensor(q, dtype=torch.float32)

        P, R = tk.forward(t_root_pos, t_root_rot, t_q)
        P_fast, R_fast = ftk.forward(t_root_pos, t_root_rot, t_q)

        diff_fast = float((P - P_fast).abs().max())
        if diff_fast > max_fast_diff:
            max_fast_diff = diff_fast

        P_np = P.numpy()
        R_np = R.numpy()

        for b_idx, b_name in enumerate(tk.body_names):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b_name)
            mj_pos = d.xpos[bid]
            mj_mat = d.xmat[bid].reshape(3, 3)
            pos_err = float(np.max(np.abs(P_np[b_idx] - mj_pos)))
            rot_err = float(np.max(np.abs(R_np[b_idx] - mj_mat)))
            if pos_err > max_body_pos_err:
                max_body_pos_err = pos_err
            if rot_err > max_body_rot_err:
                max_body_rot_err = rot_err

        # Check sites
        Ps, Rs = tk.forward_sites(P, R, site_indices)
        Ps_np = Ps.numpy()
        Rs_np = Rs.numpy()

        for i, sid in enumerate(site_indices):
            mj_spos = d.site_xpos[sid]
            mj_smat = d.site_xmat[sid].reshape(3, 3)
            spos_err = float(np.max(np.abs(Ps_np[i] - mj_spos)))
            srot_err = float(np.max(np.abs(Rs_np[i] - mj_smat)))
            if spos_err > max_site_pos_err:
                max_site_pos_err = spos_err
            if srot_err > max_site_rot_err:
                max_site_rot_err = srot_err

    print(f"\nFK Parity Summary over {num_poses} poses:")
    print(f"  Max Body Pos Error: {max_body_pos_err:.3e} m (gate: < 1e-5)")
    print(f"  Max Body Rot Error: {max_body_rot_err:.3e} rad (gate: < 1e-4)")
    print(f"  Max Site Pos Error: {max_site_pos_err:.3e} m (gate: < 1e-5)")
    print(f"  Max Site Rot Error: {max_site_rot_err:.3e} rad (gate: < 1e-4)")
    print(f"  Max FastTree Diff:  {max_fast_diff:.3e} m")

    assert max_body_pos_err < 1e-5, f"Body pos error {max_body_pos_err} exceeded gate 1e-5"
    assert max_body_rot_err < 1e-4, f"Body rot error {max_body_rot_err} exceeded gate 1e-4"
    assert max_site_pos_err < 1e-5, f"Site pos error {max_site_pos_err} exceeded gate 1e-5"
    assert max_site_rot_err < 1e-4, f"Site rot error {max_site_rot_err} exceeded gate 1e-4"
    assert max_fast_diff < 1e-5, f"FastTreeKinematics mismatch {max_fast_diff}"


def test_nao_kinematics_oracle_parity(mj_model):
    """Verify end-effector positions match third_party/nao_kinematics in Torso frame (< 0.1 mm)."""
    try:
        from Python.NAOH25.nao_kinematics import NAOKinematics, CHAIN_L_LEG, CHAIN_R_LEG, CHAIN_L_ARM, CHAIN_R_ARM
    except ImportError:
        from NAOH25.nao_kinematics import NAOKinematics, CHAIN_L_LEG, CHAIN_R_LEG, CHAIN_L_ARM, CHAIN_R_ARM

    m = mj_model
    d = mujoco.MjData(m)
    nk = NAOKinematics()

    # Zero pose check
    mujoco.mj_kinematics(m, d)
    bid_torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # Left foot
    nk.set_chain(CHAIN_L_LEG, [0.0] * 6)
    T_lfoot = nk.get_forward_effector(CHAIN_L_LEG)
    p_lfoot_dh = T_lfoot.transform(np.zeros(3)) / 1000.0
    sid_lankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "l_ankle_track")
    p_lsole_mj = d.site_xpos[sid_lankle] - d.xpos[bid_torso]
    err_lfoot = float(np.linalg.norm(p_lfoot_dh - p_lsole_mj))
    assert err_lfoot < 1e-4, f"Left foot offset mismatch: {err_lfoot} m"

    # Right foot
    nk.set_chain(CHAIN_R_LEG, [0.0] * 6)
    T_rfoot = nk.get_forward_effector(CHAIN_R_LEG)
    p_rfoot_dh = T_rfoot.transform(np.zeros(3)) / 1000.0
    sid_rankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_ankle_track")
    p_rsole_mj = d.site_xpos[sid_rankle] - d.xpos[bid_torso]
    err_rfoot = float(np.linalg.norm(p_rfoot_dh - p_rsole_mj))
    assert err_rfoot < 1e-4, f"Right foot offset mismatch: {err_rfoot} m"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
