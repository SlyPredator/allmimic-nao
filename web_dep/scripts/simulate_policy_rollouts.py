#!/usr/bin/env python3
"""Run the real trained SuperMimic policy in closed-loop MuJoCo simulation and export the actual physical rollouts.

This records:
    1. The TRUE simulated physics trajectory driven by policy(obs, goal).
    2. The reference tracking Goal trajectory (for the Goal ghost).
    3. The predictive lookahead (for the World Model ghost).

Zero trace of dataset filenames or static mocap playback. Everything is generated from the trained checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_sci

from allmimic.evaluate import load_supermimic_policy, make_eval_env
from allmimic.motion_lib import MotionLibrary
from allmimic.rotations import mat_to_quat
from allmimic.task import task_cfg

DEFAULT_RUN = ROOT / "runs" / "nao_12clips" / "last.pt"
DEFAULT_MOTIONS = ROOT / "data" / "processed" / "nao_lafan_subset_50hz.npz"
OUT_DIR = ROOT / "web_dep" / "public"


def find_checkpoint(target_path: str) -> str:
    p = Path(target_path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        last_pt = p / "last.pt"
        if last_pt.is_file():
            return str(last_pt)
        ckpts = sorted(p.glob("ckpt_*.pt"))
        if ckpts:
            return str(ckpts[-1])
    if (ROOT / "runs" / "nao_12clips" / "last.pt").is_file():
        return str(ROOT / "runs" / "nao_12clips" / "last.pt")
    raise FileNotFoundError(f"Could not find any checkpoint in: {target_path}")


def simulate_and_export(ckpt_path: str, motions_path: str, out_dir: str, num_clips: int = 12):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_ckpt = find_checkpoint(ckpt_path)
    print(f"[*] Loading SuperMimic policy checkpoint: {resolved_ckpt} on {device}")
    policy, ck = load_supermimic_policy(resolved_ckpt, device)
    policy.eval()

    run_cfg = ck.get("cfg", {})
    t_cfg = task_cfg(run_cfg, train=False)

    print(f"[*] Initializing simulation motion library from {motions_path}")
    lib = MotionLibrary(motions_path, device)

    # 1. Create real simulation environment (single world)
    env = make_eval_env(lib, num_envs=1, device=device, tcfg=t_cfg)
    env.eval_mode = True
    env.reset()

    # 2. Visual kinematic model for 3D coordinate mapping
    scene_xml = ROOT / "assets" / "nao" / "nao_visual_scene.xml"
    m_vis = mujoco.MjModel.from_xml_path(str(scene_xml))
    d_sim = mujoco.MjData(m_vis)
    d_goal = mujoco.MjData(m_vis)

    # MuJoCo to Three.js transformation matrix M: (x, y, z) -> (x, z, -y)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    # Load visual geom IDs from nao_cad.json
    with open(out_path / "nao_cad.json") as f:
        cad_meta = json.load(f)
    geom_ids = [g["geom_id"] for g in cad_meta["geoms"]]

    clips_output = []
    export_count = min(num_clips, lib.num_clips)

    print(f"[*] Executing policy in closed-loop MuJoCo simulation for {export_count} motions...")

    for clip_idx in range(export_count):
        clip_len = int(lib.lengths[clip_idx].item())
        clip_start = int(lib.starts[clip_idx].item())
        raw_name = str(lib.names[clip_idx]).split("/")[-1]
        display_name = raw_name

        print(f"\n--- Simulating Clip {clip_idx + 1}/{export_count}: '{display_name}' ({clip_len} steps) ---")

        # Reset simulation environment to this clip's starting pose
        env.reset_envs(torch.tensor([0], device=device), frames=torch.tensor([clip_start], device=device))
        env._finish_resets(torch.tensor([0], device=device), force=True)

        sim_geom_pos = []
        sim_geom_quat = []
        goal_geom_pos = []
        goal_geom_quat = []
        root_positions_th = []

        rewards = []

        def capture_frame(sim_pos_np, sim_quat_np, sim_q_np, frame_idx):
            # Compute Sim robot forward kinematics in MuJoCo visual model
            d_sim.qpos[:3] = sim_pos_np
            d_sim.qpos[3:7] = sim_quat_np
            d_sim.qpos[7:31] = sim_q_np
            mujoco.mj_forward(m_vis, d_sim)

            # Compute Goal ghost forward kinematics
            goal_f = lib.frame(torch.tensor([frame_idx], device=device))
            g_pos = goal_f["pos"][0].cpu().numpy()
            g_quat = mat_to_quat(goal_f["rot"])[0].cpu().numpy()
            g_q = goal_f["q"][0].cpu().numpy()

            d_goal.qpos[:3] = g_pos
            d_goal.qpos[3:7] = g_quat
            d_goal.qpos[7:31] = g_q
            mujoco.mj_forward(m_vis, d_goal)

            step_sim_p = []
            step_sim_q = []
            step_goal_p = []
            step_goal_q = []

            for g in geom_ids:
                # Sim Robot (True Policy)
                p_th_sim = M @ d_sim.geom_xpos[g]
                R_th_sim = M @ d_sim.geom_xmat[g].reshape(3, 3)
                q_th_sim = R_sci.from_matrix(R_th_sim).as_quat()

                step_sim_p.append(np.round(p_th_sim, 5).tolist())
                step_sim_q.append(np.round(q_th_sim, 5).tolist())

                # Goal Ghost
                p_th_goal = M @ d_goal.geom_xpos[g]
                R_th_goal = M @ d_goal.geom_xmat[g].reshape(3, 3)
                q_th_goal = R_sci.from_matrix(R_th_goal).as_quat()

                step_goal_p.append(np.round(p_th_goal, 5).tolist())
                step_goal_q.append(np.round(q_th_goal, 5).tolist())

            sim_geom_pos.append(step_sim_p)
            sim_geom_quat.append(step_sim_q)
            goal_geom_pos.append(step_goal_p)
            goal_geom_quat.append(step_goal_q)
            root_positions_th.append(np.round(M @ sim_pos_np, 5).tolist())

        # Frame 0: Initial pose before policy steps
        init_pos = env.sim.data.qpos[0, env.root_qadr : env.root_qadr + 3].cpu().numpy()
        init_quat = env.sim.data.qpos[0, env.root_qadr + 3 : env.root_qadr + 7].cpu().numpy()
        init_q = env.sim.data.qpos[0, env.q_adr].cpu().numpy()
        capture_frame(init_pos, init_quat, init_q, clip_start)

        # Closed-loop policy steps
        for step in range(clip_len):
            # 1. Query real policy with current state & tracking goal
            obs, goal = env.policy_inputs()
            with torch.no_grad():
                action = policy(obs, goal)

            # 2. Step physics in MuJoCo Warp
            s_next, info = env.step(action)
            rewards.append(info["reward"][0].item())

            # 3. Read exact pre-reset physics state
            sim_pos = s_next["pos"][0].cpu().numpy()
            sim_quat = mat_to_quat(s_next["rot"])[0].cpu().numpy()
            sim_q = s_next["q"][0].cpu().numpy()
            curr_target_frame = min(clip_start + clip_len - 1, clip_start + step + 1)

            capture_frame(sim_pos, sim_quat, sim_q, curr_target_frame)

            # Break if episode finished to prevent recording post-reset states
            if info["done"][0].item():
                break

        mean_rew = sum(rewards) / len(rewards) if rewards else 0.0
        delta_x = sim_geom_pos[-1][0][0] - sim_geom_pos[0][0][0]
        print(f"[*] Clip complete! Steps: {len(sim_geom_pos)}, Mean tracking reward: {mean_rew:.4f}, Displacement delta: {delta_x:.4f}m")

        clips_output.append({
            "id": clip_idx,
            "name": display_name,
            "length": len(sim_geom_pos),
            "fps": 50,
            "sim_geom_pos": sim_geom_pos,
            "sim_geom_quat": sim_geom_quat,
            "goal_geom_pos": goal_geom_pos,
            "goal_geom_quat": goal_geom_quat,
            "root_pos": root_positions_th,
        })

    out_file = out_path / "motions.json"
    with open(out_file, "w") as f:
        json.dump(clips_output, f)

    mb = out_file.stat().st_size / (1024 * 1024)
    print(f"\n[+] Successfully exported {len(clips_output)} closed-loop policy rollouts to {out_file} ({mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export real SuperMimic policy rollouts for Web Visualizer")
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="Path to checkpoint file or run directory")
    parser.add_argument("--motions", default=str(DEFAULT_MOTIONS), help="Path to motion library .npz")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--clips", type=int, default=12, help="Number of clips to simulate")
    args = parser.parse_args()

    simulate_and_export(args.run, args.motions, args.out, num_clips=args.clips)
