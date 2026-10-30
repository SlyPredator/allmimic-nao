#!/usr/bin/env python3
"""Interactive 3D Stick-Figure Humanoid Visualizer for BONES-SEED Motion Clips.

Uses a minimalist, research-paper stick-figure kinematic humanoid (cylindrical capsule bones,
glossy spherical joints, clean studio lighting, soft floor reflections).

Usage:
    python reference_motions/animate_bones_seed.py
    python reference_motions/animate_bones_seed.py --clip shadow_boxing
    python reference_motions/animate_bones_seed.py --clip vertical_mountain_climber
    python reference_motions/animate_bones_seed.py --clip warm_up_legs
    python reference_motions/animate_bones_seed.py --clip 0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

REPR_DIR = Path(__file__).resolve().parent
XML_PATH = REPR_DIR / "stick_humanoid.xml"
DATA_DIR = REPR_DIR / "data_bones_seed" / "g1"


def get_clip_list() -> list[Path]:
    csvs = sorted(DATA_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No BONES-SEED CSV clips found in {DATA_DIR}!")
    return csvs


def load_clip_data(csv_path: Path) -> np.ndarray:
    """Load BONES-SEED CSV, converting translation from cm->m, Euler rot->quat, and deg->rad."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    # Root position (cm -> meters)
    root_pos = data[:, 1:4] / 100.0
    # Root rotation (Euler xyz in degrees -> quaternion [x, y, z, w])
    euler_deg = data[:, 4:7]
    r = R.from_euler("xyz", euler_deg, degrees=True)
    root_quat_xyzw = r.as_quat()
    # 29 joint DOFs (degrees -> radians)
    joints_rad = np.deg2rad(data[:, 7:36])
    return np.hstack([root_pos, root_quat_xyzw, joints_rad])


def main():
    parser = argparse.ArgumentParser(description="Research Stick-Figure Humanoid Animator (BONES-SEED Motions)")
    parser.add_argument("--clip", default=0, help="Initial clip index or clip name search term (e.g. shadow_boxing)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback framerate (default: 30 fps)")
    parser.add_argument("--xml", default=str(XML_PATH), help="Path to stick figure MuJoCo XML")
    args = parser.parse_args()

    clip_paths = get_clip_list()
    num_clips = len(clip_paths)
    print(f"Loaded {num_clips} BONES-SEED motion clips from {DATA_DIR}")

    # Clip index resolution
    try:
        curr_idx = int(args.clip)
    except ValueError:
        clip_names = [p.stem for p in clip_paths]
        matching = [i for i, name in enumerate(clip_names) if str(args.clip).lower() in name.lower()]
        curr_idx = matching[0] if matching else 0
    curr_idx = max(0, min(curr_idx, num_clips - 1))

    # Load MuJoCo Model
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    state = {
        "clip_idx": curr_idx,
        "clip_data": None,
        "clip_name": "",
        "num_frames": 0,
        "frame_idx": 0,
        "paused": False,
        "speed": 1.0,
        "reset_needed": True,
    }

    def load_current_clip():
        target_path = clip_paths[state["clip_idx"]]
        state["clip_name"] = target_path.stem
        state["clip_data"] = load_clip_data(target_path)
        state["num_frames"] = state["clip_data"].shape[0]
        state["frame_idx"] = 0
        state["reset_needed"] = False
        duration_s = state["num_frames"] / args.fps
        print(f"\n>> Playing Clip [{state['clip_idx'] + 1}/{num_clips}]: '{state['clip_name']}' "
              f"({state['num_frames']} frames, {duration_s:.1f}s)")

    load_current_clip()

    def key_callback(keycode: int):
        # Space (32): Pause / Resume
        if keycode == 32:
            state["paused"] = not state["paused"]
            print(f" [{'PAUSED' if state['paused'] else 'RESUMED'}]")
        # 'R' or 'r': Reset clip
        elif keycode in (82, 114):
            state["frame_idx"] = 0
            print(" [RESET TO FRAME 0]")
        # ']' (93): Next clip
        elif keycode == 93:
            state["clip_idx"] = (state["clip_idx"] + 1) % num_clips
            state["reset_needed"] = True
        # '[' (91): Previous clip
        elif keycode == 91:
            state["clip_idx"] = (state["clip_idx"] - 1) % num_clips
            state["reset_needed"] = True
        # Right Arrow (262): Step forward 15 frames
        elif keycode == 262:
            state["frame_idx"] = min(state["num_frames"] - 1, state["frame_idx"] + 15)
            print(f" [FRAME -> {state['frame_idx']}/{state['num_frames']}]")
        # Left Arrow (263): Step backward 15 frames
        elif keycode == 263:
            state["frame_idx"] = max(0, state["frame_idx"] - 15)
            print(f" [FRAME -> {state['frame_idx']}/{state['num_frames']}]")
        # Up Arrow (265): Speed up
        elif keycode == 265:
            state["speed"] = min(3.0, round(state["speed"] + 0.25, 2))
            print(f" [PLAYBACK SPEED: {state['speed']:.2f}x]")
        # Down Arrow (264): Slow down
        elif keycode == 264:
            state["speed"] = max(0.25, round(state["speed"] - 0.25, 2))
            print(f" [PLAYBACK SPEED: {state['speed']:.2f}x]")

    print("\n------------------ CONTROLS ------------------")
    print("  [SPACE]        : Pause / Resume playback")
    print("  [R]            : Restart current clip")
    print("  [ ] / [        : Next / Previous clip")
    print("  [->] / [<-]    : Step +/- 15 frames")
    print("  [^] / [v]      : Increase / Decrease playback speed")
    print("  Mouse Drag     : Orbit, Pan & Zoom camera")
    print("----------------------------------------------\n")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -16.0
        viewer.cam.azimuth = 135.0

        target_dt = 1.0 / args.fps

        while viewer.is_running():
            t_start = time.time()

            if state["reset_needed"]:
                load_current_clip()

            k = state["frame_idx"]
            frame = state["clip_data"][k]

            # 36 columns: pos(3), quat_xyzw(4), dof(29) -> qpos: pos(3), quat_wxyz(4), dof(29)
            d.qpos[:3] = frame[:3]
            d.qpos[3:7] = frame[[6, 3, 4, 5]]  # convert xyzw to MuJoCo wxyz
            d.qpos[7:36] = frame[7:36]

            mujoco.mj_forward(m, d)
            viewer.sync()

            if not state["paused"]:
                state["frame_idx"] += 1
                if state["frame_idx"] >= state["num_frames"]:
                    print(f"  Clip '{state['clip_name']}' completed. Looping...")
                    state["frame_idx"] = 0

            dt_step = target_dt / state["speed"]
            elapsed = time.time() - t_start
            sleep_time = dt_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
