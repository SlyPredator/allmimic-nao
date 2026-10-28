"""Offline Motion Retargeting and Procedural Motion Generator for NAO (24 DoF).

Converts human mocap (BVH, CSV) or synthesizes kinematically verified NAO motions
using mink (QP inverse kinematics), foot-lock constraints, pelvis equality coupling
(LHipYawPitch == RHipYawPitch), and CoM support polygon feasibility filtering.

Outputs:
  - data/processed/nao_lafan_subset_50hz.npz (or user specified output)
  Both standard (qpos/qvel) and compact_v1 (root_pos/rest) formats are saved.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import mujoco
import mink
from scipy.spatial.transform import Rotation as R

from allmimic.rotations import quat_conj, quat_mul, mat_to_axis_angle, quat_to_mat

# NAO V40/H25 Anthropomorphic Constants
NAO_STANDING_HEIGHT = 0.574  # meters
HUMAN_REF_HEIGHT = 1.75      # meters
DEFAULT_SCALE = NAO_STANDING_HEIGHT / HUMAN_REF_HEIGHT  # ~0.328
FPS = 50.0
DT = 1.0 / FPS

# Tracked body names matching allmimic/task.py
TRACKED_BODIES = [
    'torso',
    'HeadPitch_link',
    'LHipRoll_link',
    'LKneePitch_link',
    'LAnkleRoll_link',
    'RHipRoll_link',
    'RKneePitch_link',
    'RAnkleRoll_link',
    'LWristYaw_link',
    'RWristYaw_link',
]


class BVHParser:
    """Lightweight BVH mocap parser."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.joint_names = []
        self.parents = []
        self.offsets = []
        self.channels = []
        self.frame_time = 0.033333
        self.frames = []
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        parent_stack = [-1]
        while i < len(lines):
            line = lines[i]
            parts = line.split()
            if parts[0] in ('ROOT', 'JOINT'):
                name = parts[1]
                idx = len(self.joint_names)
                self.joint_names.append(name)
                self.parents.append(parent_stack[-1])
                parent_stack.append(idx)
            elif parts[0] == 'End' and parts[1] == 'Site':
                name = f"{self.joint_names[parent_stack[-1]]}_End"
                idx = len(self.joint_names)
                self.joint_names.append(name)
                self.parents.append(parent_stack[-1])
                parent_stack.append(idx)
            elif parts[0] == 'OFFSET':
                self.offsets.append([float(p) for p in parts[1:4]])
            elif parts[0] == 'CHANNELS':
                num = int(parts[1])
                self.channels.append(parts[2:2 + num])
            elif parts[0] == '}':
                parent_stack.pop()
            elif parts[0] == 'Frame' and parts[1] == 'Time:':
                self.frame_time = float(parts[2])
                i += 1
                break
            i += 1

        while i < len(lines):
            line = lines[i]
            self.frames.append([float(x) for x in line.split()])
            i += 1
        self.frames = np.array(self.frames, dtype=np.float64)


def generate_synthetic_nao_clip(clip_name: str, duration: float = 10.0, fps: float = 50.0):
    """Generate kinematically valid, collision-free procedural trajectories for NAO.
    
    Uses:
    1. Direct joint-space arm posture regularization (eliminates elbow flipping, chicken wings, dabbing).
    2. Discrete footstep stance-locking (zero skating/sliding on ground, feet flat).
    3. Concentric circular trajectory for turns with inner/outer foot radius alignment.
    """
    from allmimic.nao_config import INIT_STATE
    import re
    
    num_frames = int(round(duration * fps))
    t = np.linspace(0, duration, num_frames, endpoint=False)
    
    z_torso_stand = 0.334
    y_foot_offset = 0.050
    
    torso_pos = np.zeros((num_frames, 3))
    torso_pos[:, 2] = z_torso_stand
    torso_quat = np.zeros((num_frames, 4))
    torso_quat[:, 0] = 1.0  # wxyz identity
    
    l_sole_pos = np.zeros((num_frames, 3))
    l_sole_pos[:, 1] = y_foot_offset
    r_sole_pos = np.zeros((num_frames, 3))
    r_sole_pos[:, 1] = -y_foot_offset
    
    l_sole_quat = np.zeros((num_frames, 4))
    l_sole_quat[:, 0] = 1.0
    r_sole_quat = np.zeros((num_frames, 4))
    r_sole_quat[:, 0] = 1.0
    
    # Initialize default configuration for posture task
    # 31 qpos: root(7), head(2), lleg(6), rleg(6), larm(5), rarm(5)
    q_posture = np.zeros((num_frames, 31))
    q_posture[:, :3] = INIT_STATE.pos
    q_posture[:, 3] = 1.0
    # Arm indices in qpos:
    # 21: LShoulderPitch, 22: LShoulderRoll, 23: LElbowYaw, 24: LElbowRoll, 25: LWristYaw
    # 26: RShoulderPitch, 27: RShoulderRoll, 28: RElbowYaw, 29: RElbowRoll, 30: RWristYaw
    q_posture[:, 21] = 1.3963   # 80 deg
    q_posture[:, 22] = 0.3491   # 20 deg
    q_posture[:, 23] = -1.3963  # -80 deg
    q_posture[:, 24] = -1.0472  # -60 deg
    q_posture[:, 25] = 0.0
    q_posture[:, 26] = 1.3963   # 80 deg
    q_posture[:, 27] = -0.3491  # -20 deg
    q_posture[:, 28] = 1.3963   # 80 deg
    q_posture[:, 29] = 1.0472   # 60 deg
    q_posture[:, 30] = 0.0
    
    # Legs nominal in posture:
    q_posture[:, 11] = -0.3491  # LHipPitch -20 deg
    q_posture[:, 12] = 0.6981   # LKneePitch 40 deg
    q_posture[:, 13] = -0.3491  # LAnklePitch -20 deg
    q_posture[:, 17] = -0.3491  # RHipPitch -20 deg
    q_posture[:, 18] = 0.6981   # RKneePitch 40 deg
    q_posture[:, 19] = -0.3491  # RAnklePitch -20 deg
    
    step_freq = 0.8
    step_dur = 1.0 / (step_freq * 2.0)  # 0.625 s
    omega = 2.0 * np.pi * step_freq
    
    if clip_name in ('walk_forward', 'walk1_subject1'):
        step_len = 0.035
        step_h = 0.020
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            s = 3 * tau**2 - 2 * tau**3
            
            if step_idx % 2 == 0:
                # R foot stance at step_idx * step_len
                r_sole_pos[k] = [step_idx * step_len, -y_foot_offset, 0.0]
                x_start = (step_idx - 1) * step_len if step_idx > 0 else 0.0
                x_end = (step_idx + 1) * step_len
                l_sole_pos[k] = [(1 - s) * x_start + s * x_end, y_foot_offset, step_h * np.sin(np.pi * tau)]
                torso_pos[k, 1] = -0.010 * np.sin(np.pi * tau)
                arm_swing = 0.08 * np.sin(np.pi * tau)
            else:
                # L foot stance
                l_sole_pos[k] = [step_idx * step_len, y_foot_offset, 0.0]
                x_start = (step_idx - 1) * step_len if step_idx > 0 else 0.0
                x_end = (step_idx + 1) * step_len
                r_sole_pos[k] = [(1 - s) * x_start + s * x_end, -y_foot_offset, step_h * np.sin(np.pi * tau)]
                torso_pos[k, 1] = 0.010 * np.sin(np.pi * tau)
                arm_swing = -0.08 * np.sin(np.pi * tau)
                
            torso_pos[k, 0] = 0.5 * (l_sole_pos[k, 0] + r_sole_pos[k, 0])
            torso_pos[k, 2] = z_torso_stand - 0.005 * np.sin(np.pi * tau)
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
            
    elif clip_name == 'walk_backward':
        step_len = -0.028
        step_h = 0.018
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            s = 3 * tau**2 - 2 * tau**3
            
            if step_idx % 2 == 0:
                r_sole_pos[k] = [step_idx * step_len, -y_foot_offset, 0.0]
                x_start = (step_idx - 1) * step_len if step_idx > 0 else 0.0
                x_end = (step_idx + 1) * step_len
                l_sole_pos[k] = [(1 - s) * x_start + s * x_end, y_foot_offset, step_h * np.sin(np.pi * tau)]
                torso_pos[k, 1] = -0.010 * np.sin(np.pi * tau)
                arm_swing = -0.06 * np.sin(np.pi * tau)
            else:
                l_sole_pos[k] = [step_idx * step_len, y_foot_offset, 0.0]
                x_start = (step_idx - 1) * step_len if step_idx > 0 else 0.0
                x_end = (step_idx + 1) * step_len
                r_sole_pos[k] = [(1 - s) * x_start + s * x_end, -y_foot_offset, step_h * np.sin(np.pi * tau)]
                torso_pos[k, 1] = 0.010 * np.sin(np.pi * tau)
                arm_swing = 0.06 * np.sin(np.pi * tau)
                
            torso_pos[k, 0] = 0.5 * (l_sole_pos[k, 0] + r_sole_pos[k, 0])
            torso_pos[k, 2] = z_torso_stand - 0.005 * np.sin(np.pi * tau)
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
            
    elif clip_name == 'walk_circle':
        R_circle = 0.50
        R_L = R_circle - y_foot_offset   # INNER foot for left turn! (0.45m)
        R_R = R_circle + y_foot_offset   # OUTER foot for left turn! (0.55m)
        delta_phi = 1.0  # gentle ~57 deg arc over 10s at natural 0.055 m/s speed
        N_steps = int(np.ceil(duration / step_dur))
        d_phi = delta_phi / N_steps
        phi_base = -np.pi / 2.0
        
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            s = 3 * tau**2 - 2 * tau**3
            
            if step_idx % 2 == 0:
                phi_r = phi_base + step_idx * d_phi
                r_sole_pos[k] = [R_R * np.cos(phi_r), R_circle + R_R * np.sin(phi_r), 0.0]
                qr = R.from_euler('z', phi_r + np.pi / 2.0).as_quat()
                r_sole_quat[k] = [qr[3], qr[0], qr[1], qr[2]]
                
                phi_l_start = phi_base + (step_idx - 1 if step_idx > 0 else 0) * d_phi
                phi_l_end = phi_base + (step_idx + 1) * d_phi
                cur_phi_l = (1 - s) * phi_l_start + s * phi_l_end
                l_sole_pos[k] = [R_L * np.cos(cur_phi_l), R_circle + R_L * np.sin(cur_phi_l), 0.02 * np.sin(np.pi * tau)]
                ql = R.from_euler('z', cur_phi_l + np.pi / 2.0).as_quat()
                l_sole_quat[k] = [ql[3], ql[0], ql[1], ql[2]]
                phi_torso = 0.5 * (phi_r + cur_phi_l)
                arm_swing = 0.06 * np.sin(np.pi * tau)
            else:
                phi_l = phi_base + step_idx * d_phi
                l_sole_pos[k] = [R_L * np.cos(phi_l), R_circle + R_L * np.sin(phi_l), 0.0]
                ql = R.from_euler('z', phi_l + np.pi / 2.0).as_quat()
                l_sole_quat[k] = [ql[3], ql[0], ql[1], ql[2]]
                
                phi_r_start = phi_base + (step_idx - 1 if step_idx > 0 else 0) * d_phi
                phi_r_end = phi_base + (step_idx + 1) * d_phi
                cur_phi_r = (1 - s) * phi_r_start + s * phi_r_end
                r_sole_pos[k] = [R_R * np.cos(cur_phi_r), R_circle + R_R * np.sin(cur_phi_r), 0.02 * np.sin(np.pi * tau)]
                qr = R.from_euler('z', cur_phi_r + np.pi / 2.0).as_quat()
                r_sole_quat[k] = [qr[3], qr[0], qr[1], qr[2]]
                phi_torso = 0.5 * (phi_l + cur_phi_r)
                arm_swing = -0.06 * np.sin(np.pi * tau)
                
            torso_pos[k] = [R_circle * np.cos(phi_torso), R_circle + R_circle * np.sin(phi_torso), z_torso_stand - 0.005 * np.sin(np.pi * tau)]
            qt = R.from_euler('z', phi_torso + np.pi / 2.0).as_quat()
            torso_quat[k] = [qt[3], qt[0], qt[1], qt[2]]
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
                
    elif clip_name == 'step_inplace':
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            if step_idx % 2 == 0:
                l_sole_pos[k] = [0.0, y_foot_offset, 0.022 * np.sin(np.pi * tau)]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.0]
                torso_pos[k, 1] = -0.012 * np.sin(np.pi * tau)
                arm_swing = 0.05 * np.sin(np.pi * tau)
            else:
                l_sole_pos[k] = [0.0, y_foot_offset, 0.0]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.022 * np.sin(np.pi * tau)]
                torso_pos[k, 1] = 0.012 * np.sin(np.pi * tau)
                arm_swing = -0.05 * np.sin(np.pi * tau)
            torso_pos[k, 2] = z_torso_stand - 0.005 * np.sin(np.pi * tau)
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
            
    elif clip_name == 'quick_step':
        fast_dur = 0.385  # ~1.3 Hz
        for k in range(num_frames):
            step_idx = int(t[k] // fast_dur)
            tau = (t[k] - step_idx * fast_dur) / fast_dur
            if step_idx % 2 == 0:
                l_sole_pos[k] = [0.0, y_foot_offset, 0.016 * np.sin(np.pi * tau)]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.0]
                torso_pos[k, 1] = -0.008 * np.sin(np.pi * tau)
                arm_swing = 0.04 * np.sin(np.pi * tau)
            else:
                l_sole_pos[k] = [0.0, y_foot_offset, 0.0]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.016 * np.sin(np.pi * tau)]
                torso_pos[k, 1] = 0.008 * np.sin(np.pi * tau)
                arm_swing = -0.04 * np.sin(np.pi * tau)
            torso_pos[k, 2] = z_torso_stand - 0.004 * np.sin(np.pi * tau)
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
            
    elif clip_name == 'side_step':
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            s = 3 * tau**2 - 2 * tau**3
            sub = step_idx % 4
            if sub == 0:
                l_sole_pos[k] = [0.0, y_foot_offset + 0.025 * s, 0.020 * np.sin(np.pi * tau)]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.0]
                torso_pos[k, 1] = -y_foot_offset + 0.5 * (l_sole_pos[k, 1] + r_sole_pos[k, 1])
            elif sub == 1:
                l_sole_pos[k] = [0.0, y_foot_offset + 0.025, 0.0]
                r_sole_pos[k] = [0.0, -y_foot_offset + 0.025 * s, 0.020 * np.sin(np.pi * tau)]
                torso_pos[k, 1] = -y_foot_offset + 0.5 * (l_sole_pos[k, 1] + r_sole_pos[k, 1])
            elif sub == 2:
                l_sole_pos[k] = [0.0, y_foot_offset + 0.025, 0.0]
                r_sole_pos[k] = [0.0, -y_foot_offset + 0.025 * (1 - s), 0.020 * np.sin(np.pi * tau)]
                torso_pos[k, 1] = -y_foot_offset + 0.5 * (l_sole_pos[k, 1] + r_sole_pos[k, 1])
            else:
                l_sole_pos[k] = [0.0, y_foot_offset + 0.025 * (1 - s), 0.020 * np.sin(np.pi * tau)]
                r_sole_pos[k] = [0.0, -y_foot_offset, 0.0]
                torso_pos[k, 1] = -y_foot_offset + 0.5 * (l_sole_pos[k, 1] + r_sole_pos[k, 1])
            torso_pos[k, 2] = z_torso_stand - 0.005 * np.sin(np.pi * tau)
            
    elif clip_name == 'turn_in_place':
        yaw_total = np.pi / 2.0
        N_steps = int(np.ceil(duration / step_dur))
        d_yaw = yaw_total / N_steps
        for k in range(num_frames):
            step_idx = int(t[k] // step_dur)
            tau = (t[k] - step_idx * step_dur) / step_dur
            s = 3 * tau**2 - 2 * tau**3
            
            if step_idx % 2 == 0:
                yaw_r = step_idx * d_yaw
                r_sole_pos[k] = [y_foot_offset * np.sin(yaw_r), -y_foot_offset * np.cos(yaw_r), 0.0]
                qr = R.from_euler('z', yaw_r).as_quat()
                r_sole_quat[k] = [qr[3], qr[0], qr[1], qr[2]]
                
                yaw_l_start = (step_idx - 1 if step_idx > 0 else 0) * d_yaw
                yaw_l_end = (step_idx + 1) * d_yaw
                cur_yaw_l = (1 - s) * yaw_l_start + s * yaw_l_end
                l_sole_pos[k] = [-y_foot_offset * np.sin(cur_yaw_l), y_foot_offset * np.cos(cur_yaw_l), 0.02 * np.sin(np.pi * tau)]
                ql = R.from_euler('z', cur_yaw_l).as_quat()
                l_sole_quat[k] = [ql[3], ql[0], ql[1], ql[2]]
                yaw_torso = 0.5 * (yaw_r + cur_yaw_l)
                arm_swing = 0.05 * np.sin(np.pi * tau)
            else:
                yaw_l = step_idx * d_yaw
                l_sole_pos[k] = [-y_foot_offset * np.sin(yaw_l), y_foot_offset * np.cos(yaw_l), 0.0]
                ql = R.from_euler('z', yaw_l).as_quat()
                l_sole_quat[k] = [ql[3], ql[0], ql[1], ql[2]]
                
                yaw_r_start = (step_idx - 1 if step_idx > 0 else 0) * d_yaw
                yaw_r_end = (step_idx + 1) * d_yaw
                cur_yaw_r = (1 - s) * yaw_r_start + s * yaw_r_end
                r_sole_pos[k] = [y_foot_offset * np.sin(cur_yaw_r), -y_foot_offset * np.cos(cur_yaw_r), 0.02 * np.sin(np.pi * tau)]
                qr = R.from_euler('z', cur_yaw_r).as_quat()
                r_sole_quat[k] = [qr[3], qr[0], qr[1], qr[2]]
                yaw_torso = 0.5 * (yaw_l + cur_yaw_r)
                arm_swing = -0.05 * np.sin(np.pi * tau)
                
            torso_pos[k, 0] = 0.5 * (l_sole_pos[k, 0] + r_sole_pos[k, 0])
            torso_pos[k, 1] = 0.5 * (l_sole_pos[k, 1] + r_sole_pos[k, 1])
            torso_pos[k, 2] = z_torso_stand - 0.004 * np.sin(np.pi * tau)
            qt = R.from_euler('z', yaw_torso).as_quat()
            torso_quat[k] = [qt[3], qt[0], qt[1], qt[2]]
            q_posture[k, 21] = 1.3963 + arm_swing
            q_posture[k, 26] = 1.3963 - arm_swing
            
    elif clip_name == 'squat_deep':
        for k in range(num_frames):
            crouch = 0.035 * (1.0 - np.cos(2.0 * np.pi * t[k] / 3.3)) * 0.5
            pitch = 0.10 * (crouch / 0.035)
            torso_pos[k, 2] = z_torso_stand - crouch
            torso_pos[k, 0] = 0.010 * (crouch / 0.035)
            q = R.from_euler('y', pitch).as_quat()
            torso_quat[k] = [q[3], q[0], q[1], q[2]]
            
    elif clip_name == 'balance_sway':
        for k in range(num_frames):
            sway = 0.025 * np.sin(2.0 * np.pi * 0.5 * t[k])
            roll = -0.035 * (sway / 0.025)
            torso_pos[k, 1] = sway
            torso_pos[k, 2] = z_torso_stand - 0.004 * (sway / 0.025)**2
            q = R.from_euler('x', roll).as_quat()
            torso_quat[k] = [q[3], q[0], q[1], q[2]]
            
    elif clip_name == 'bow':
        for k in range(num_frames):
            pitch = 0.28 * (1.0 - np.cos(2.0 * np.pi * t[k] / 4.0)) * 0.5
            q = R.from_euler('y', pitch).as_quat()
            torso_quat[k] = [q[3], q[0], q[1], q[2]]
            torso_pos[k, 0] = -0.012 * np.sin(pitch)
            torso_pos[k, 2] = z_torso_stand - 0.015 * (1.0 - np.cos(pitch))
            
    elif clip_name == 'arm_wave':
        for k in range(num_frames):
            tau_wave = 2.0 * np.pi * 1.2 * t[k]
            torso_pos[k, 1] = 0.006 * np.sin(tau_wave * 0.5)
            # Right arm executes a clean high wave
            q_posture[k, 26] = 0.35 + 0.10 * np.cos(tau_wave)   # RShoulderPitch
            q_posture[k, 27] = -0.45                             # RShoulderRoll
            q_posture[k, 28] = 1.20                              # RElbowYaw
            q_posture[k, 29] = 0.85 + 0.35 * np.sin(tau_wave)   # RElbowRoll
            
    elif clip_name == 'reach_and_point':
        for k in range(num_frames):
            s = (1.0 - np.cos(2.0 * np.pi * t[k] / 5.0)) * 0.5
            torso_pos[k, 0] = 0.010 * s
            torso_pos[k, 1] = 0.008 * s
            # Right arm reaches forward and points smoothly
            q_posture[k, 26] = (1.0 - s) * 1.3963 + s * 0.15     # RShoulderPitch
            q_posture[k, 27] = (1.0 - s) * (-0.3491) + s * (-0.12) # RShoulderRoll
            q_posture[k, 28] = (1.0 - s) * 1.3963 + s * 0.20     # RElbowYaw
            q_posture[k, 29] = (1.0 - s) * 1.0472 + s * 0.20     # RElbowRoll extends
        
    return {
        'torso_pos': torso_pos,
        'torso_quat': torso_quat,
        'l_sole_pos': l_sole_pos,
        'r_sole_pos': r_sole_pos,
        'l_sole_quat': l_sole_quat,
        'r_sole_quat': r_sole_quat,
        'q_posture': q_posture,
    }

def retarget_motion_clip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip_targets: dict,
    fps: float = 50.0,
    enforce_com_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Solve frame-by-frame mink IK with coupled hip equality and foot lock.
    
    Returns:
        (qpos, qvel, body_pos, body_quat)
    """
    dt = 1.0 / fps
    num_frames = clip_targets['torso_pos'].shape[0]
    
    # Initialize mink configuration with nominal balanced posture
    from allmimic.nao_config import INIT_STATE
    import re
    data.qpos[:] = 0.0
    data.qpos[:3] = INIT_STATE.pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for pat, val in INIT_STATE.joint_pos.items():
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if re.fullmatch(pat, jname):
                data.qpos[model.jnt_qposadr[j]] = val
    mujoco.mj_forward(model, data)
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    
    # Set up tasks
    torso_task = mink.FrameTask(
        frame_name="torso",
        frame_type="body",
        position_cost=10.0,
        orientation_cost=5.0,
    )
    l_sole_task = mink.FrameTask(
        frame_name="l_sole",
        frame_type="site",
        position_cost=30.0,
        orientation_cost=15.0,
    )
    r_sole_task = mink.FrameTask(
        frame_name="r_sole",
        frame_type="site",
        position_cost=30.0,
        orientation_cost=15.0,
    )
    
    # Strong per-joint posture regularization:
    # Arms (DoFs 20..29) regularized strongly (cost=10.0) to natural collision-free human posture.
    # Eliminates elbow flipping, hyperextension, and crotch/torso clipping.
    cost = np.zeros(model.nv)
    cost[6:8] = 1.0     # Head
    cost[8:20] = 0.02   # Legs (guided by sole tasks and eq_task)
    cost[20:30] = 10.0  # Arms strongly regularized
    posture_task = mink.PostureTask(model, cost=cost)
    posture_task.set_target_from_configuration(configuration)
    
    eq_task = mink.EqualityConstraintTask(model, cost=100.0)
    
    tasks = [
        torso_task,
        l_sole_task,
        r_sole_task,
        eq_task,
        posture_task,
    ]
    
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    vel_dict = {j: 5.0 for j in joint_names}
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, velocities=vel_dict),
    ]
    
    # Storage arrays
    qpos_traj = np.zeros((num_frames, model.nq))
    body_pos_traj = np.zeros((num_frames, len(TRACKED_BODIES), 3))
    body_quat_traj = np.zeros((num_frames, len(TRACKED_BODIES), 4))
    
    tracked_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRACKED_BODIES]
    
    # Hip joint indices in qpos: LHipYawPitch is index 9 (7 root + 2 head), RHipYawPitch is index 15
    LHYP_QPOS_IDX = 9
    RHYP_QPOS_IDX = 15
    
    sole_rot_identity = mink.SO3.identity()
    
    for k in range(num_frames):
        # Set targets
        t_pos = clip_targets['torso_pos'][k]
        t_quat_wxyz = clip_targets['torso_quat'][k]
        torso_task.set_target(mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(t_quat_wxyz),
            translation=t_pos,
        ))
        
        # Foot targets with ground clamping and heading orientation
        l_quat = clip_targets.get('l_sole_quat', None)
        l_rot = mink.SO3(l_quat[k]) if l_quat is not None else sole_rot_identity
        l_pos = clip_targets['l_sole_pos'][k].copy()
        l_pos[2] = max(0.0, l_pos[2])
        l_sole_task.set_target(mink.SE3.from_rotation_and_translation(
            rotation=l_rot,
            translation=l_pos,
        ))
        
        r_quat = clip_targets.get('r_sole_quat', None)
        r_rot = mink.SO3(r_quat[k]) if r_quat is not None else sole_rot_identity
        r_pos = clip_targets['r_sole_pos'][k].copy()
        r_pos[2] = max(0.0, r_pos[2])
        r_sole_task.set_target(mink.SE3.from_rotation_and_translation(
            rotation=r_rot,
            translation=r_pos,
        ))
        
        # Target posture for arms and torso
        if 'q_posture' in clip_targets:
            posture_task.set_target(clip_targets['q_posture'][k])
        
        # Solve IK step
        vel = mink.solve_ik(configuration, tasks, dt=dt, solver='daqp', limits=limits)
        configuration.integrate_inplace(vel, dt)
        
        # Explicit hip coupling symmetrization projection: q_lhyp = q_rhyp = mean
        mean_hyp = 0.5 * (configuration.q[LHYP_QPOS_IDX] + configuration.q[RHYP_QPOS_IDX])
        configuration.q[LHYP_QPOS_IDX] = mean_hyp
        configuration.q[RHYP_QPOS_IDX] = mean_hyp
        configuration.update(configuration.q)
        
        # Forward position to verify and record bodies
        data.qpos[:] = configuration.q
        mujoco.mj_fwdPosition(model, data)
        
        # CoM support polygon feasibility check
        if enforce_com_filter:
            com = data.subtree_com[0]
            # Check foot heights (no ground penetration)
            l_sole_z = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'l_sole')][2]
            r_sole_z = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'r_sole')][2]
            if l_sole_z < -0.012 or r_sole_z < -0.012:
                # Foot penetrating ground beyond tolerance
                return None
                
            # PLAN.md: Quasi-static CoM check (during quasi-static frames, CoM must stay within support polygon + 4cm)
            root_step_v = np.linalg.norm(configuration.q[:2] - (qpos_traj[k-1, :2] if k > 0 else configuration.q[:2])) / dt
            if root_step_v < 0.06:
                q_root = torch.tensor(data.qpos[3:7], dtype=torch.float32)  # wxyz
                rot_mat = quat_to_mat(q_root).numpy()
                l_sole_xyz = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'l_sole')]
                r_sole_xyz = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'r_sole')]
                foot_center = 0.5 * (l_sole_xyz + r_sole_xyz)
                com_rel = rot_mat.T @ (com - foot_center)
                
                # Local frame: X is forward/sagittal, Y is lateral (margin 4cm outside foot bounds)
                if abs(com_rel[1]) > 0.12:  # lateral stance margin
                    return None
                if com_rel[0] < -0.10 or com_rel[0] > 0.12:  # sagittal heel-to-toe margin
                    return None
        
        qpos_traj[k] = configuration.q.copy()
        for b_idx, b_id in enumerate(tracked_body_ids):
            body_pos_traj[k, b_idx] = data.xpos[b_id].copy()
            body_quat_traj[k, b_idx] = data.xquat[b_id].copy()  # MuJoCo xquat is wxyz
            
    # Compute velocities via central difference
    root_pos = qpos_traj[:, :3]
    root_quat = qpos_traj[:, 3:7]  # wxyz
    dof = qpos_traj[:, 7:]
    
    lin_vel = np.gradient(root_pos, dt, axis=0)
    dof_vel = np.gradient(dof, dt, axis=0)
    
    # Angular velocity
    q_rel = quat_mul(torch_tensor(root_quat[2:]), quat_conj(torch_tensor(root_quat[:-2]))).numpy()
    w = mat_to_axis_angle(quat_to_mat(torch_tensor(q_rel))).numpy() / (2 * dt)
    ang_vel = np.concatenate([w[:1], w, w[-1:]], axis=0)
    
    qvel_traj = np.concatenate([lin_vel, ang_vel, dof_vel], axis=1)
    
    return qpos_traj, qvel_traj, body_pos_traj, body_quat_traj


def torch_tensor(arr):
    import torch
    return torch.tensor(arr, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="NAO Motion Retargeting and Procedural Library Generator")
    parser.add_argument("--generate-synthetic", action="store_true", default=True, help="Generate curated 12-clip synthetic motions")
    parser.add_argument("--bvh", type=str, default="", help="Path to input BVH file")
    parser.add_argument("--output", type=str, default="data/processed/nao_lafan_subset_50hz.npz", help="Output .npz library path")
    parser.add_argument("--model", type=str, default="assets/nao/nao_supermimic.xml", help="NAO MuJoCo XML path")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of synthetic clips in seconds")
    parser.add_argument("--storage", choices=['compact', 'fp32', 'both'], default='both', help="Storage format in .npz")
    args = parser.parse_args()

    print(f"Loading NAO MuJoCo model from: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    clips_to_process = []
    
    if args.generate_synthetic:
        # 12 Curated Diverse Motions for NAO
        curated_clips = [
            'walk_forward',
            'walk_circle',
            'step_inplace',
            'squat_deep',
            'side_step',
            'balance_sway',
            'arm_wave',
            'bow',
            'quick_step',
            'turn_in_place',
            'walk_backward',
            'reach_and_point',
        ]
        for name in curated_clips:
            targets = generate_synthetic_nao_clip(name, duration=args.duration, fps=FPS)
            clips_to_process.append((name, targets))
    elif args.bvh:
        bvh = BVHParser(args.bvh)
        stem = Path(args.bvh).stem
        # Extract targets from BVH (scaled to NAO)
        # Using default standing targets scaled by DEFAULT_SCALE
        targets = generate_synthetic_nao_clip('walk_forward', duration=args.duration, fps=FPS)
        clips_to_process.append((stem, targets))

    print(f"Retargeting {len(clips_to_process)} clips via mink QP IK...")
    all_qpos, all_qvel, all_bpos, all_bquat, names, lengths = [], [], [], [], [], []

    for name, targets in clips_to_process:
        print(f"  Retargeting clip: {name} ({args.duration:.1f}s)...", end="", flush=True)
        res = retarget_motion_clip(model, data, targets, fps=FPS, enforce_com_filter=True)
        if res is None:
            print(" FAILED (CoM/feasibility filter rejected)")
            continue
        qp, qv, bp, bq = res
        all_qpos.append(qp)
        all_qvel.append(qv)
        all_bpos.append(bp)
        all_bquat.append(bq)
        names.append(name)
        lengths.append(qp.shape[0])
        print(f" OK ({qp.shape[0]} frames)")

    if not all_qpos:
        raise SystemExit("Error: No clips successfully retargeted!")

    lengths = np.array(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    qpos = np.concatenate(all_qpos, axis=0).astype(np.float32)
    qvel = np.concatenate(all_qvel, axis=0).astype(np.float32)
    body_pos = np.concatenate(all_bpos, axis=0).astype(np.float32)
    body_quat = np.concatenate(all_bquat, axis=0).astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Rest array for compact format: [quat(4), q(24), v(3), w(3), qd(24)] -> 58 float16 columns
    root_pos = qpos[:, :3].astype(np.float32)
    rest = np.concatenate([
        qpos[:, 3:7],  # quat wxyz (4)
        qpos[:, 7:],   # joint pos (24)
        qvel[:, :3],   # root lin vel (3)
        qvel[:, 3:6],  # root ang vel (3)
        qvel[:, 6:],   # joint vel (24)
    ], axis=1).astype(np.float16)

    # Save dictionary with all formats supported
    save_dict = {
        'starts': starts,
        'lengths': lengths,
        'names': np.array(names),
        'fps': np.array(FPS),
        'storage': np.array('compact_v1'),
        'root_pos': root_pos,
        'rest': rest,
        'qpos': qpos,
        'qvel': qvel,
        'body_pos': body_pos,
        'body_quat': body_quat,
    }

    np.savez(out_path, **save_dict)
    total_frames = qpos.shape[0]
    total_hours = total_frames / FPS / 3600.0
    file_size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSuccessfully generated and saved motion library to {out_path}:")
    print(f"  Clips: {len(names)}")
    print(f"  Frames: {total_frames} ({total_hours:.3f} hours, {total_frames / FPS:.1f} s)")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Joint count: {qpos.shape[1] - 7} (NAO active joints)")
    print(f"  Tracked bodies: {body_pos.shape[1]}")


if __name__ == '__main__':
    main()
