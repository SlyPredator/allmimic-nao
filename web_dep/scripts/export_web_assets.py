#!/usr/bin/env python3
"""Export trained SuperMimic / allmimic policies and reference motions for the web deployment.

Usage:
    /home/robotics/miniconda3/envs/supermimic_nao/bin/python web_dep/scripts/export_web_assets.py \
        --run runs/nao_walk_forward \
        --out-dir web_dep/public
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import mujoco

from allmimic.evaluate import load_supermimic_policy
from allmimic.motion_lib import MotionLibrary
from allmimic.task import GOAL_FRAME_DIM, NJ, NU, OBS_DIM, XI_DIM, TaskCfg, task_cfg

DEFAULT_RUN = ROOT / "runs" / "nao_walk_forward"
DEFAULT_MOTIONS = ROOT / "data" / "processed" / "nao_lafan_subset_50hz.npz"


class PolicyExportWrapper(nn.Module):
    """Clean wrapper that takes raw unnormalized (obs, goal) and returns actions."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # Policy handles internal obs_norm and goal_norm
        return self.policy(obs, goal)


def export_assets(run_path: str, motions_path: str, out_dir: str, num_clips_export: int = 5):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    run_p = Path(run_path)
    if run_p.is_dir():
        if (run_p / "last.pt").is_file():
            run_p = run_p / "last.pt"
        else:
            ckpts = sorted(run_p.glob("ckpt_*.pt"))
            if ckpts:
                run_p = ckpts[-1]
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {run_path}")

    print(f"[*] Loading checkpoint from: {run_p}")
    policy, ck = load_supermimic_policy(str(run_p), device=device)
    policy.eval()

    run_cfg = ck.get("cfg", {})
    t_cfg = task_cfg(run_cfg, train=False)

    obs_dim = OBS_DIM * t_cfg.history
    goal_dim = GOAL_FRAME_DIM * t_cfg.future
    print(f"[*] Task specs: history={t_cfg.history} (obs_dim={obs_dim}), future={t_cfg.future} (goal_dim={goal_dim})")

    # 1. Export Policy to ONNX
    onnx_file = out_path / "policy.onnx"
    wrapper = PolicyExportWrapper(policy).to(device)
    wrapper.eval()

    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    dummy_goal = torch.randn(1, goal_dim, dtype=torch.float32, device=device)

    print(f"[*] Exporting policy to ONNX: {onnx_file}")
    with torch.no_grad():
        test_action = wrapper(dummy_obs, dummy_goal)

        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_goal),
            str(onnx_file),
            input_names=["obs", "goal"],
            output_names=["action"],
            dynamic_axes=None,  # Fixed batch size = 1 for fastest browser inference
            opset_version=14,
        )
    print(f"[+] Policy exported successfully ({onnx_file.stat().st_size / 1024:.1f} KB)")

    # 2. Export Configuration JSON
    # Joint names for NAO (standard 24 joints in nao_supermimic.xml)
    joint_names = [
        "HeadYaw", "HeadPitch",
        "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
        "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
        "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
        "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"
    ]

    config_data = {
        "model_name": "NAO SuperMimic Whole-Body Motion Tracker",
        "obs_dim": int(obs_dim),
        "goal_dim": int(goal_dim),
        "history": int(t_cfg.history),
        "future": int(t_cfg.future),
        "action_dim": int(NU),
        "action_scale": float(run_cfg.get("res_scale", 0.5)),
        "dt": float(t_cfg.dt),
        "control_dt": 0.02,  # 50 Hz control policy
        "joint_names": joint_names,
        "obs_norm_mean": policy.obs_norm.mean.cpu().tolist(),
        "obs_norm_std": policy.obs_norm.std.cpu().tolist(),
        "goal_norm_mean": policy.goal_norm.mean.cpu().tolist(),
        "goal_norm_std": policy.goal_norm.std.cpu().tolist(),
    }

    config_file = out_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"[+] Config exported to: {config_file}")

    # 3. Export Parity Test Vector
    parity_data = {
        "obs": dummy_obs.squeeze(0).tolist(),
        "goal": dummy_goal.squeeze(0).tolist(),
        "expected_action": test_action.squeeze(0).tolist(),
    }
    parity_file = out_path / "parity_test.json"
    with open(parity_file, "w") as f:
        json.dump(parity_data, f, indent=2)
    print(f"[+] Parity test vector saved to: {parity_file}")

    # 4. Export Reference Motion Clips with full CAD forward kinematics
    if Path(motions_path).is_file():
        from scipy.spatial.transform import Rotation as R_sci

        print(f"[*] Processing motion clips with MuJoCo kinematics from: {motions_path}")
        lib = MotionLibrary(motions_path, device="cpu")
        
        # Load visual model for forward kinematics
        scene_xml = ROOT / "assets" / "nao" / "nao_visual_scene.xml"
        m_vis = mujoco.MjModel.from_xml_path(str(scene_xml))
        d_vis = mujoco.MjData(m_vis)

        # MuJoCo to Three.js transformation matrix
        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

        # Read geom IDs from nao_cad.json
        cad_json_path = out_path / "nao_cad.json"
        with open(cad_json_path) as f:
            cad_meta = json.load(f)
        geom_ids = [g["geom_id"] for g in cad_meta["geoms"]]

        clips_data = []
        export_count = min(num_clips_export, lib.num_clips)

        for i in range(export_count):
            start = int(lib.starts[i].item())
            length = int(lib.lengths[i].item())
            end = start + length
            name = lib.names[i] if hasattr(lib, "names") and i < len(lib.names) else f"clip_{i}"

            root_pos_arr = lib.root_pos[start:end].cpu().numpy()
            root_quat_arr = lib.root_quat[start:end].cpu().numpy()
            q_arr = lib.q[start:end].cpu().numpy()

            clip_geom_pos = []
            clip_geom_quat = []
            clip_root_pos_th = []

            for f in range(length):
                d_vis.qpos[:3] = root_pos_arr[f]
                d_vis.qpos[3:7] = root_quat_arr[f]
                d_vis.qpos[7:7 + len(q_arr[f])] = q_arr[f]
                mujoco.mj_forward(m_vis, d_vis)

                frame_p = []
                frame_q = []
                for g in geom_ids:
                    p_mj = d_vis.geom_xpos[g]
                    R_mj = d_vis.geom_xmat[g].reshape(3, 3)

                    p_th = M @ p_mj
                    R_th = M @ R_mj
                    q_th = R_sci.from_matrix(R_th).as_quat()  # [x, y, z, w]

                    frame_p.append(np.round(p_th, 5).tolist())
                    frame_q.append(np.round(q_th, 5).tolist())

                clip_geom_pos.append(frame_p)
                clip_geom_quat.append(frame_q)
                
                # Transform root pos to Three.js coordinates
                clip_root_pos_th.append(np.round(M @ root_pos_arr[f], 5).tolist())

            clips_data.append({
                "id": i,
                "name": name,
                "length": length,
                "fps": 50,
                "geom_pos": clip_geom_pos,
                "geom_quat": clip_geom_quat,
                "sim_geom_pos": clip_geom_pos,
                "goal_geom_pos": clip_geom_pos,
                "root_pos": clip_root_pos_th,
            })

        motions_file = out_path / "motions.json"
        with open(motions_file, "w") as f:
            json.dump(clips_data, f)
        print(f"[+] Exported {len(clips_data)} motion clips with exact CAD geom poses to: {motions_file} ({motions_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[!] Motion library {motions_path} not found, skipping clip export.")


def main():
    parser = argparse.ArgumentParser(description="Export SuperMimic web deployment assets")
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="Run directory or checkpoint path")
    parser.add_argument("--motions", default=str(DEFAULT_MOTIONS), help="Path to motion library .npz")
    parser.add_argument("--out-dir", default="web_dep/public", help="Output directory for web assets")
    parser.add_argument("--num-clips", type=int, default=5, help="Number of clips to export")
    args = parser.parse_args()

    export_assets(args.run, args.motions, args.out_dir, args.num_clips)


if __name__ == "__main__":
    main()
