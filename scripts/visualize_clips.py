"""Visualize and record all clips in nao_lafan_subset_50hz.npz to MP4 videos."""

import os
from pathlib import Path
import numpy as np
import mujoco
import mediapy as media

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "assets" / "nao" / "nao_supermimic.xml"
DATA_PATH = ROOT / "data" / "processed" / "nao_lafan_subset_50hz.npz"
VIDEOS_DIR = ROOT / "videos"

def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=480, width=640)
    
    # Custom camera settings: tracking or nice orbit view
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    camera.distance = 1.6
    camera.elevation = -18.0
    camera.azimuth = 135.0

    data = np.load(str(DATA_PATH))
    names = data['names']
    starts = data['starts']
    lengths = data['lengths']
    qpos_all = data['qpos']
    fps = float(data.get('fps', 50.0))

    print(f"Loaded {len(names)} clips. Rendering to {VIDEOS_DIR} at {fps} FPS...")

    for i, name in enumerate(names):
        start = int(starts[i])
        length = int(lengths[i])
        clip_qpos = qpos_all[start : start + length]
        
        frames = []
        for f in range(length):
            d.qpos[:] = clip_qpos[f]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera)
            img = renderer.render()
            frames.append(img)
            
        out_file = VIDEOS_DIR / f"{name}.mp4"
        media.write_video(str(out_file), frames, fps=fps)
        print(f"[{i+1}/{len(names)}] Rendered {name} ({length} frames) -> {out_file.name}")

    print("All videos rendered successfully!")

if __name__ == "__main__":
    main()

