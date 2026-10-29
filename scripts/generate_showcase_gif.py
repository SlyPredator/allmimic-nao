"""Generate a multi-clip showcase GIF/video for SuperMimic NAO policies.

Renders a synchronized side-by-side grid of all motion capabilities
with live physics simulation, ghost reference overlay, and tracking cameras.

Usage:
    python scripts/generate_showcase_gif.py --run runs/nao_12clips --duration 4.0
    python scripts/generate_showcase_gif.py --run runs/nao_full --out assets/nao_showcase.gif --save-mp4
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
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from allmimic.evaluate import load_supermimic_policy, make_eval_env
from allmimic.motion_lib import MotionLibrary
from allmimic.rotations import mat_to_quat
from allmimic.task import task_cfg

SCENE_PATH = ROOT / "assets" / "nao" / "nao_visual_scene.xml"
DEFAULT_MOTIONS = ROOT / "data" / "processed" / "nao_lafan_subset_50hz.npz"
DEFAULT_OUT = ROOT / "assets" / "nao_challenge_cases.gif"


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


def get_fonts():
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    bold_path = font_paths[0] if os.path.isfile(font_paths[0]) else None
    reg_path = font_paths[1] if os.path.isfile(font_paths[1]) else None

    if bold_path and reg_path:
        return {
            "title": ImageFont.truetype(bold_path, 13),
            "regular": ImageFont.truetype(reg_path, 10),
            "bold": ImageFont.truetype(bold_path, 10),
            "badge": ImageFont.truetype(reg_path, 9),
        }
    else:
        default_font = ImageFont.load_default()
        return {
            "title": default_font,
            "regular": default_font,
            "bold": default_font,
            "badge": default_font,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Motion Showcase GIF for NAO")
    parser.add_argument("--run", default="runs/nao_12clips", help="Path to checkpoint file or run directory")
    parser.add_argument("--motions", default=str(DEFAULT_MOTIONS), help="Path to motion library .npz")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for showcase .gif")
    parser.add_argument("--save-mp4", action="store_true", help="Also save an .mp4 video alongside the gif")
    parser.add_argument("--duration", type=float, default=4.0, help="Total recording duration in seconds (default: 4.0s)")
    parser.add_argument("--fps", type=int, default=15, help="Output GIF/video framerate (default: 15 fps)")
    parser.add_argument("--cols", type=int, default=4, help="Number of grid columns (default: 4)")
    parser.add_argument("--tile-w", type=int, default=180, help="Width of each simulation tile in pixels (default: 180)")
    parser.add_argument("--tile-h", type=int, default=144, help="Height of each simulation tile in pixels (default: 144)")
    parser.add_argument("--no-ghost", action="store_true", help="Disable blue reference ghost overlay")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device
    ckpt_path = find_checkpoint(args.run)
    run_name = Path(ckpt_path).parent.name if Path(ckpt_path).name == "last.pt" else Path(ckpt_path).stem
    print(f">> Loading policy: {ckpt_path}")

    # 1. Load Policy and checkpoint metadata
    policy, ck = load_supermimic_policy(ckpt_path, device)
    run_cfg = ck.get("cfg", {})
    t_cfg = task_cfg(run_cfg, train=False)

    # 2. Load Motion Library
    lib = MotionLibrary(args.motions, device)
    num_clips = lib.num_clips
    cols = args.cols
    rows = (num_clips + cols - 1) // cols
    print(f">> Loaded {num_clips} motions from {args.motions}. Grid layout: {rows}x{cols}")

    # 3. Create parallel simulation environments (one per motion clip)
    env = make_eval_env(lib, num_envs=num_clips, device=device, tcfg=t_cfg)
    env.eval_mode = True
    env.reset()

    # Reset env i to clip i
    env.reset_envs(torch.arange(num_clips, device=device), frames=lib.starts[:num_clips])
    env._finish_resets(torch.arange(num_clips, device=device), force=True)

    # 4. Offscreen MuJoCo visualizer setup
    m_vis = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    d_vis = mujoco.MjData(m_vis)
    d_ref = mujoco.MjData(m_vis)

    tile_w, tile_h = args.tile_w, args.tile_h
    renderer = mujoco.Renderer(m_vis, tile_h, tile_w)
    cam = mujoco.MjvCamera()
    cam.distance = 1.6
    cam.elevation = -14
    cam.azimuth = 135

    GHOST_RGBA = [0.18, 0.55, 1.0, 0.40]
    IGNORE_NAMES = {"Finger", "Thumb"}
    vis_geoms = [
        i for i in range(m_vis.ngeom)
        if m_vis.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and not any(k in (mujoco.mj_id2name(m_vis, mujoco.mjtObj.mjOBJ_GEOM, i) or "") for k in IGNORE_NAMES)
    ]

    fonts = get_fonts()

    # Layout dimensions (rounded up to multiple of 16 for video codec compatibility)
    gap = 2
    header_h = 38
    footer_h = 46
    raw_w = cols * tile_w + (cols - 1) * gap
    raw_h = header_h + rows * tile_h + (rows - 1) * gap + footer_h
    canvas_w = ((raw_w + 15) // 16) * 16
    canvas_h = ((raw_h + 15) // 16) * 16
    grid_h = rows * tile_h + (rows - 1) * gap
    # Center grid horizontally if canvas was padded
    x_offset = (canvas_w - raw_w) // 2

    sim_dt = 0.02  # 50 Hz native simulation
    gif_dt = 1.0 / args.fps
    steps_per_frame = max(1, round(gif_dt / sim_dt))
    total_gif_frames = int(args.duration * args.fps)

    print(f">> Generating {total_gif_frames} frames ({args.duration:.1f}s @ {args.fps} fps)...")
    t_start_total = time.time()

    frames = []
    active_mask = [True] * num_clips

    for frame_idx in range(total_gif_frames):
        # Step physics forward steps_per_frame times
        for _ in range(steps_per_frame):
            obs, goal = env.policy_inputs()
            with torch.no_grad():
                a = policy(obs, goal)
                s_next, info = env.step(a)

            # Check terminations
            for i in range(num_clips):
                if info["terminated"][i].item():
                    active_mask[i] = False
                if info["done"][i].item() and not info["terminated"][i].item():
                    # Clip completed normally, loop back
                    env.reset_envs(torch.tensor([i], device=device), frames=torch.tensor([int(lib.starts[i])], device=device))
                    env._finish_resets(torch.tensor([i], device=device), force=True)

        current_time = (frame_idx + 1) * gif_dt

        # Render all tiles for current frame
        tile_images = []
        for i in range(num_clips):
            sim_pos = env.sim.data.qpos[i, env.root_qadr : env.root_qadr + 3].cpu().numpy()
            sim_quat = env.sim.data.qpos[i, env.root_qadr + 3 : env.root_qadr + 7].cpu().numpy()
            sim_q = env.sim.data.qpos[i, env.q_adr].cpu().numpy()

            d_vis.qpos[:3] = sim_pos
            d_vis.qpos[3:7] = sim_quat
            d_vis.qpos[7:31] = sim_q
            mujoco.mj_forward(m_vis, d_vis)

            # Reference pose
            if not args.no_ghost:
                curr_frame = env.frame[i].unsqueeze(0)
                goal_f = lib.frame(curr_frame)
                g_pos = goal_f["pos"][0].cpu().numpy()
                g_quat = mat_to_quat(goal_f["rot"])[0].cpu().numpy()
                g_q = goal_f["q"][0].cpu().numpy()

                d_ref.qpos[:3] = g_pos
                d_ref.qpos[3:7] = g_quat
                d_ref.qpos[7:31] = g_q
                mujoco.mj_kinematics(m_vis, d_ref)

            # Camera smoothly tracks robot x, y
            cam.lookat = [sim_pos[0], sim_pos[1], 0.32]
            renderer.update_scene(d_vis, camera=cam)
            scn = renderer.scene

            # Append ghost geoms
            if not args.no_ghost:
                for gi in vis_geoms:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        m_vis.geom_type[gi],
                        m_vis.geom_size[gi],
                        d_ref.geom_xpos[gi],
                        d_ref.geom_xmat[gi],
                        GHOST_RGBA,
                    )
                    g.dataid = 2 * m_vis.geom_dataid[gi]
                    g.objtype = mujoco.mjtObj.mjOBJ_GEOM
                    g.objid = gi
                    scn.ngeom += 1

            tile_img = Image.fromarray(renderer.render())
            draw_t = ImageDraw.Draw(tile_img)

            # Draw clip label badge
            clean_name = lib.names[i].replace("_", " ")
            badge_bg = (20, 25, 35, 210) if active_mask[i] else (160, 30, 30, 220)
            badge_text = clean_name if active_mask[i] else f"{clean_name} (fall)"
            badge_w = int(draw_t.textlength(badge_text, font=fonts["badge"])) + 10
            draw_t.rectangle([(4, 4), (4 + badge_w, 18)], fill=badge_bg)
            draw_t.text((8, 5), badge_text, fill=(245, 248, 255), font=fonts["badge"])

            tile_images.append(tile_img)

        # Assemble full composite image
        canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)

        # Top Header
        draw.text((16, 11), "nao mimic", fill=(35, 40, 50), font=fonts["title"])
        draw.text((105, 11), "·", fill=(160, 160, 160), font=fonts["title"])
        draw.text((120, 13), f"Motion Capabilities Matrix ({num_clips} Clips in Real-Time Physics)", fill=(95, 100, 112), font=fonts["regular"])

        # Grid viewports
        for idx, t_img in enumerate(tile_images):
            r = idx // cols
            c = idx % cols
            x = x_offset + c * (tile_w + gap)
            y = header_h + r * (tile_h + gap)
            canvas.paste(t_img, (x, y))

        # Bottom Footer
        f_y = header_h + grid_h
        draw.line([(0, header_h - 1), (canvas_w, header_h - 1)], fill=(220, 224, 230), width=1)
        draw.line([(0, f_y), (canvas_w, f_y)], fill=(220, 224, 230), width=1)

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        draw.text((16, f_y + 8), f"Policy: {run_name} · 12 Motions · {gpu_name}", fill=(30, 35, 45), font=fonts["bold"])
        draw.text((16, f_y + 24), "Silver: simulated robot", fill=(90, 95, 105), font=fonts["regular"])
        if not args.no_ghost:
            draw.text((150, f_y + 24), "Blue: reference motion", fill=(25, 110, 220), font=fonts["regular"])
        draw.text((290, f_y + 24), f"t = {current_time:.1f} s", fill=(100, 105, 115), font=fonts["regular"])

        # Right status
        num_active = sum(active_mask)
        status_txt = f"{num_active}/{num_clips} active"
        status_color = (20, 130, 60) if num_active == num_clips else (200, 90, 20)
        sw = int(draw.textlength(status_txt, font=fonts["bold"]))
        draw.text((canvas_w - 16 - sw, f_y + 8), status_txt, fill=status_color, font=fonts["bold"])

        tag_txt = "SuperMimic NAO"
        tw = int(draw.textlength(tag_txt, font=fonts["regular"]))
        draw.text((canvas_w - 16 - tw, f_y + 24), tag_txt, fill=(120, 125, 135), font=fonts["regular"])

        frames.append(canvas)
        if (frame_idx + 1) % 15 == 0 or (frame_idx + 1) == total_gif_frames:
            print(f"  Rendered frame {frame_idx + 1}/{total_gif_frames} ({(time.time() - t_start_total):.1f}s)")

    # 5. Save GIF
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f">> Saving animated GIF to {out_path} ...")
    duration_ms = int(1000.0 / args.fps)

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f">> GIF saved successfully: {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")

    # Optional MP4 save
    if args.save_mp4:
        mp4_path = out_path.with_suffix(".mp4")
        print(f">> Saving MP4 video to {mp4_path} ...")
        try:
            import imageio
            writer = imageio.get_writer(str(mp4_path), fps=args.fps, quality=8)
            for f in frames:
                writer.append_data(np.array(f))
            writer.close()
            print(f">> MP4 saved successfully: {mp4_path} ({os.path.getsize(mp4_path) / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  Warning: could not save MP4: {e}")


if __name__ == "__main__":
    main()
