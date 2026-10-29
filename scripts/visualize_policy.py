"""Interactive 3D MuJoCo Visualizer for SuperMimic NAO policies with Goal & World Model ghosts.

Usage:
    conda run -n supermimic_nao python scripts/visualize_policy.py --run runs/nao_smoke_stageA
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import torch

from allmimic.evaluate import load_supermimic_policy, make_eval_env
from allmimic.motion_lib import MotionLibrary
from allmimic.networks import WorldModel
from allmimic.rotations import mat_to_quat, quat_to_mat
from allmimic.task import XI_DIM, state_to_flat, task_cfg

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT / "assets" / "nao" / "nao_visual_scene.xml"
MODEL_PATH = SCENE_PATH if SCENE_PATH.is_file() else ROOT / "assets" / "nao" / "nao_supermimic.xml"
DEFAULT_MOTIONS = ROOT / "data" / "processed" / "nao_lafan_subset_50hz.npz"


def find_checkpoint(target_path: str) -> str:
    if os.path.isfile(target_path):
        return target_path
    if os.path.isdir(target_path):
        last_pt = os.path.join(target_path, "last.pt")
        if os.path.isfile(last_pt):
            return last_pt
        ckpts = sorted(glob.glob(os.path.join(target_path, "ckpt_*.pt")))
        if ckpts:
            return ckpts[-1]
    raise FileNotFoundError(f"Could not find any checkpoint in: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="SuperMimic NAO Live Policy Visualizer")
    parser.add_argument("--run", default="runs/nao_smoke_stageA", help="Path to checkpoint file or run directory")
    parser.add_argument("--motions", default=str(DEFAULT_MOTIONS), help="Path to motion library .npz")
    parser.add_argument("--clip", default=0, help="Initial clip index (int) or clip name")
    parser.add_argument("--action-noise", type=float, default=0.0, help="Action exploration noise std")
    parser.add_argument("--friction", type=float, default=1.0, help="Ground friction coefficient")
    parser.add_argument("--floor-reflectance", type=float, default=0.05, help="Floor mirror reflectance (default: 0.05)")
    parser.add_argument("--ghost-goal", action="store_true", default=True, help="Render Goal ghost")
    parser.add_argument("--ghost-wm", action="store_true", default=True, help="Render World Model ghost")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max steps to run (for tests/recording)")
    args = parser.parse_args()

    device = args.device
    ckpt_path = find_checkpoint(args.run)
    print(f"Loading SuperMimic Checkpoint: {ckpt_path}")

    # 1. Load Policy and checkpoint metadata
    policy, ck = load_supermimic_policy(ckpt_path, device)
    run_cfg = ck.get("cfg", {})
    t_cfg = task_cfg(run_cfg, train=False)

    # 2. Load Motion Library
    lib = MotionLibrary(args.motions, device)
    print(f"Loaded {lib.num_clips} motion clips from {args.motions}")

    # Clip selection
    try:
        clip_id = int(args.clip)
    except ValueError:
        clip_id = lib.names.index(args.clip)
    clip_id = max(0, min(clip_id, lib.num_clips - 1))

    # 3. Create single-world simulation environment
    env = make_eval_env(lib, num_envs=1, device=device, tcfg=t_cfg)
    env.eval_mode = True

    # 4. Optional World Model loading
    wm = None
    if "wm" in ck:
        try:
            s_dummy = env.reset()
            feat_dim = env.task.state_features(s_dummy).shape[-1]
            wm = WorldModel(
                env.task,
                feat_dim,
                XI_DIM,
                norm_eps=run_cfg.get("wm_norm_eps", 1e-4),
                rot_out=run_cfg.get("wm_rot_out", "axis"),
            ).to(device)
            wm.load_state_dict(ck["wm"])
            wm.eval()
            print("World Model loaded successfully for imagination ghosting.")
        except Exception as e:
            print(f"Note: World Model loading bypassed ({e}).")

    # 5. Native MuJoCo visual model for viewer rendering
    m_vis = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mat_floor = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_mat")
    if mat_floor >= 0:
        m_vis.mat_reflectance[mat_floor] = float(args.floor_reflectance)
    d_vis = mujoco.MjData(m_vis)
    d_goal = mujoco.MjData(m_vis)
    d_wm = mujoco.MjData(m_vis)

    # Visual geom indices for ghosts: include main body visual skin (exclude individual finger/thumb mesh clutter)
    vis_geoms = []
    for i in range(m_vis.ngeom):
        if m_vis.geom_group[i] == 2:
            mid = m_vis.geom_dataid[i]
            mname = mujoco.mj_id2name(m_vis, mujoco.mjtObj.mjOBJ_MESH, mid) if mid >= 0 else ""
            if "Finger" not in mname and "Thumb" not in mname:
                vis_geoms.append(i)

    # Ghost appearance colors
    GOAL_RGBA = np.array([1.0, 1.0, 1.0, 0.35], dtype=np.float32)       # Semi-transparent white
    WM_RGBA = np.array([0.2, 0.65, 1.0, 0.50], dtype=np.float32)        # Semi-transparent blue/cyan

    # Interactive control state
    state_ctrl = {
        "paused": False,
        "reset_requested": True,
        "clip_id": clip_id,
        "show_goal": args.ghost_goal,
        "show_wm": args.ghost_wm,
        "action_noise": args.action_noise,
        "step_count": 0,
    }

    def key_callback(keycode: int):
        # Space: Pause/Resume
        if keycode == 32:
            state_ctrl["paused"] = not state_ctrl["paused"]
            print(f" [{'PAUSED' if state_ctrl['paused'] else 'RESUMED'}]")
        # 'R' or 'r': Reset
        elif keycode in (82, 114):
            state_ctrl["reset_requested"] = True
            print(" [RESET CLIP]")
        # 'G' or 'g': Toggle Goal ghost
        elif keycode in (71, 103):
            state_ctrl["show_goal"] = not state_ctrl["show_goal"]
            print(f" [Goal ghost: {'ON' if state_ctrl['show_goal'] else 'OFF'}]")
        # 'W' or 'w': Toggle World Model ghost
        elif keycode in (87, 119):
            state_ctrl["show_wm"] = not state_ctrl["show_wm"]
            print(f" [World Model ghost: {'ON' if state_ctrl['show_wm'] else 'OFF'}]")
        # ']': Next clip
        elif keycode == 93:
            state_ctrl["clip_id"] = (state_ctrl["clip_id"] + 1) % lib.num_clips
            state_ctrl["reset_requested"] = True
            print(f" [NEXT CLIP -> {lib.names[state_ctrl['clip_id']]}]")
        # '[': Previous clip
        elif keycode == 91:
            state_ctrl["clip_id"] = (state_ctrl["clip_id"] - 1) % lib.num_clips
            state_ctrl["reset_requested"] = True
            print(f" [PREV CLIP -> {lib.names[state_ctrl['clip_id']]}]")
        # '+' or '=': Increase noise
        elif keycode in (43, 61):
            state_ctrl["action_noise"] = round(state_ctrl["action_noise"] + 0.05, 2)
            print(f" [Action noise: {state_ctrl['action_noise']:.2f}]")
        # '-': Decrease noise
        elif keycode in (45, 95):
            state_ctrl["action_noise"] = max(0.0, round(state_ctrl["action_noise"] - 0.05, 2))
            print(f" [Action noise: {state_ctrl['action_noise']:.2f}]")

    print("\n------------------ CONTROLS ------------------")
    print("  [SPACE]      : Pause / Resume")
    print("  [R]          : Reset current clip")
    print("  [ ] / [      : Next / Previous motion clip")
    print("  [G]          : Toggle Goal ghost (White)")
    print("  [W]          : Toggle World Model ghost (Blue)")
    print("  [+] / [-]    : Adjust exploration action noise")
    print("  Mouse Drag   : Orbit, Pan & Zoom camera")
    print("----------------------------------------------\n")

    clip_len = int(lib.lengths[state_ctrl["clip_id"]])
    print(f"Playing Clip [{state_ctrl['clip_id'] + 1}/{lib.num_clips}]: '{lib.names[state_ctrl['clip_id']]}' ({clip_len} frames)")

    # Launch passive viewer
    with mujoco.viewer.launch_passive(m_vis, d_vis, key_callback=key_callback) as viewer:
        # Camera configuration: track NAO torso smoothly
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 135.0

        step_dt = 1.0 / 50.0  # 50 Hz control loop
        s_wm_rollout = None

        while viewer.is_running():
            t_start = time.time()

            # Handle resets
            if state_ctrl["reset_requested"]:
                c_id = state_ctrl["clip_id"]
                clip_len = int(lib.lengths[c_id])
                clip_start_frame = int(lib.starts[c_id])
                env.reset_envs(torch.tensor([0], device=device), frames=torch.tensor([clip_start_frame], device=device))
                env._finish_resets(torch.tensor([0], device=device), force=True)
                state_ctrl["reset_requested"] = False
                state_ctrl["step_count"] = 0
                s_wm_rollout = None
                print(f"Playing Clip [{c_id + 1}/{lib.num_clips}]: '{lib.names[c_id]}' ({clip_len} frames)")

            if not state_ctrl["paused"]:
                # 1. Policy step
                obs, goal = env.policy_inputs()
                with torch.no_grad():
                    a = policy(obs, goal)
                    if state_ctrl["action_noise"] > 0:
                        a = a + state_ctrl["action_noise"] * torch.randn_like(a)
                    s_next, info = env.step(a)

                state_ctrl["step_count"] += 1

                # 2. Update simulated robot visualization
                sim_pos = env.sim.data.qpos[0, env.root_qadr : env.root_qadr + 3].cpu().numpy()
                sim_quat = env.sim.data.qpos[0, env.root_qadr + 3 : env.root_qadr + 7].cpu().numpy()
                sim_q = env.sim.data.qpos[0, env.q_adr].cpu().numpy()

                d_vis.qpos[:3] = sim_pos
                d_vis.qpos[3:7] = sim_quat
                d_vis.qpos[7:31] = sim_q
                mujoco.mj_forward(m_vis, d_vis)

                # 3. Update Goal ghost
                curr_frame = env.frame[0]
                goal_f = lib.frame(curr_frame.unsqueeze(0))
                g_pos = goal_f["pos"][0].cpu().numpy()
                g_quat = mat_to_quat(goal_f["rot"])[0].cpu().numpy()
                g_q = goal_f["q"][0].cpu().numpy()

                d_goal.qpos[:3] = g_pos
                d_goal.qpos[3:7] = g_quat
                d_goal.qpos[7:31] = g_q
                mujoco.mj_kinematics(m_vis, d_goal)

                # 4. Update World Model ghost
                if wm is not None and state_ctrl["show_wm"]:
                    with torch.no_grad():
                        # One-step or multi-step autoregressive WM prediction
                        if s_wm_rollout is None or state_ctrl["step_count"] % 16 == 0:
                            s_wm_rollout = env.state  # resync every 16 steps
                        next_ref = lib.frame((curr_frame + 1).clamp(max=lib.starts[state_ctrl["clip_id"]] + clip_len - 1).unsqueeze(0))
                        a_sp = env.task.setpoint_action(a, next_ref["q"])
                        s_wm_rollout = wm.step(s_wm_rollout, a_sp, env.xi)

                    wm_pos = s_wm_rollout["pos"][0].cpu().numpy()
                    wm_quat = mat_to_quat(s_wm_rollout["rot"])[0].cpu().numpy()
                    wm_q = s_wm_rollout["q"][0].cpu().numpy()

                    d_wm.qpos[:3] = wm_pos
                    d_wm.qpos[3:7] = wm_quat
                    d_wm.qpos[7:31] = wm_q
                    mujoco.mj_kinematics(m_vis, d_wm)

                # Check if clip finished or terminated
                if info["done"][0].item():
                    term_reason = "Fall / Limit" if info["terminated"][0].item() else "Completed"
                    print(f"  Step {state_ctrl['step_count']} -> End of clip ({term_reason}). Resetting...")
                    state_ctrl["reset_requested"] = True

                if args.max_steps > 0 and state_ctrl["step_count"] >= args.max_steps:
                    print(f"Reached max_steps ({args.max_steps}). Exiting visualizer.")
                    break

            # 5. Populate user_scn with ghost geoms inside lock
            with viewer.lock():
                viewer.user_scn.ngeom = 0

                # Goal ghost (white)
                if state_ctrl["show_goal"]:
                    for gi in vis_geoms:
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                            break
                        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            m_vis.geom_type[gi],
                            m_vis.geom_size[gi],
                            d_goal.geom_xpos[gi],
                            d_goal.geom_xmat[gi],
                            GOAL_RGBA,
                        )
                        g.dataid = 2 * m_vis.geom_dataid[gi] if m_vis.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH else m_vis.geom_dataid[gi]
                        g.objtype = mujoco.mjtObj.mjOBJ_GEOM
                        g.objid = gi
                        viewer.user_scn.ngeom += 1

                # World Model ghost (blue)
                if state_ctrl["show_wm"] and wm is not None:
                    for gi in vis_geoms:
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                            break
                        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            m_vis.geom_type[gi],
                            m_vis.geom_size[gi],
                            d_wm.geom_xpos[gi],
                            d_wm.geom_xmat[gi],
                            WM_RGBA,
                        )
                        g.dataid = 2 * m_vis.geom_dataid[gi] if m_vis.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH else m_vis.geom_dataid[gi]
                        g.objtype = mujoco.mjtObj.mjOBJ_GEOM
                        g.objid = gi
                        viewer.user_scn.ngeom += 1

            viewer.sync()

            # Maintain real-time 50 Hz frame rate
            elapsed = time.time() - t_start
            sleep_time = step_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
