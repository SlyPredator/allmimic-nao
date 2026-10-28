# NAO SuperMimic Handoff Guide (RTX A4000 Training)

This document provides complete instructions for transferring, verifying, and training the **NAO SuperMimic (`allmimic-nao`)** repository on your **NVIDIA RTX A4000 (16GB VRAM)** system.

---

## 1. System Architecture & Target Specification

| Subsystem | Kinematic Joint Count (`NJ = 24`) | Policy Action Channels (`NU = 23`) | Mode |
|---|---|---|---|
| **Head** | `HeadYaw`, `HeadPitch` (2) | 2 | Residual (`q_ref + scale * a`) |
| **Left Arm** | `LShoulderPitch`, `LShoulderRoll`, `LElbowYaw`, `LElbowRoll`, `LWristYaw` (5) | 5 | Residual (`q_ref + scale * a`) |
| **Right Arm** | `RShoulderPitch`, `RShoulderRoll`, `RElbowYaw`, `RElbowRoll`, `RWristYaw` (5) | 5 | Residual (`q_ref + scale * a`) |
| **Pelvis Coupling** | `LHipYawPitch`, `RHipYawPitch` (2 joints, equality-coupled) | **1 channel** (`a_hyp`) | Absolute (`q_default + scale * a`) |
| **Left Leg** | `LHipRoll`, `LHipPitch`, `LKneePitch`, `LAnklePitch`, `LAnkleRoll` (5) | 5 | Absolute (`q_default + scale * a`) |
| **Right Leg** | `RHipRoll`, `RHipPitch`, `RKneePitch`, `RAnklePitch`, `RAnkleRoll` (5) | 5 | Absolute (`q_default + scale * a`) |
| **Hands** | Finger joints removed & welded to wrist | 0 | Rigid mesh attached to wrist |

### Core Dimensions & Scaling:
- **Kinematic joints (`NJ`)**: 24
- **Policy action channels (`NU`)**: 23
- **State vector (`SDIM`)**: 61 (`pos` 3 + `quat` 4 + `q` 24 + `v` 3 + `w` 3 + `qd` 24)
- **Observation vector (`OBS_DIM`)**: 77 (`q - q0` 24 + `qd` 24 + `gyro` 3 + `gravity` 3 + `prev_action` 23)
- **Reward Kernel Rescaling** (adapted from G1 for NAO's 0.58 m height):
  - $\sigma_{\text{pos}} = 0.08\text{ m}$ (down from G1's 0.25 m)
  - $\sigma_{\text{rot}} = 0.4\text{ rad}$
  - Termination height: `term_z = 0.18 m` (nominal torso height ~0.33–0.42 m)

---

## 2. Completed Work & Bug Fixes

1. **Stripped & Hardened Robot Model**:
   - Built `assets/nao/nao_supermimic.xml` with 24 joints, rigid hands, stiff pelvis equality coupling, and 10 tracking sites (`torso_anchor`, `head_track`, `l/r_hand_track`, `l/r_elbow_track`, `l/r_knee_track`, `l/r_ankle_track`).
   - Fixed actuator duplicate bug: stripped raw XML `<motor>` tags so `mjlab`'s `BuiltinPositionActuatorCfg` cleanly initializes exactly 24 PD actuators.

2. **Differentiable PyTorch Forward Kinematics (`allmimic/kinematics.py`)**:
   - Implemented general Rodrigues rotation exponentiation in `FastTreeKinematics` and `TreeKinematics` to natively support NAO's canted 45° hip yaw-pitch axes (`[0, 0.707107, ±0.707107]`).
   - Added `forward_sites()` for differentiable site tracking.
   - Tested across 1,000 random poses against MuJoCo CPU kinematics: **max position error $< 10^{-5}\text{ m}$**.

3. **Hip Coupling & Action Projections**:
   - Added linear expansion matrix ($P \in \mathbb{R}^{24 \times 23}$) in `ActionProjection` (`networks.py`) and `expand_actions()` (`task.py`).
   - World Model prediction symmetrization: projects $\Delta q_{\text{LHYP}}$ and $\Delta q_{\text{RHYP}}$ to their mean $\frac{1}{2}(\hat{q}_{\text{L}} + \hat{q}_{\text{R}})$ after every step, preventing drift in imagined rollouts.
   - Fixed action rate dimension alignment ($23 \text{ vs } 24$) in `task.py`.

4. **12-Clip Curated Dataset & Offline Renderings**:
   - `data/processed/nao_lafan_subset_50hz.npz` containing 12 clips (50 Hz, 10s each, compact_v1 format).
   - Rendered offscreen MuJoCo visual recordings for all 12 clips in `videos/`:
     - `walk_forward.mp4`, `walk_circle.mp4`, `step_inplace.mp4`, `squat_deep.mp4`
     - `side_step.mp4`, `balance_sway.mp4`, `arm_wave.mp4`, `bow.mp4`
     - `quick_step.mp4`, `turn_in_place.mp4`, `walk_backward.mp4`, `reach_and_point.mp4`

5. **Cross-Platform Inductor / Triton Guard**:
   - Added clean eager fallback for Windows testing while preserving full `torch.compile` / CUDA graph acceleration on Linux.

---

## 3. Transferring to the RTX A4000 System

### Option A: Via Git / Archive
Pack the workspace (excluding `.venv`):
```bash
# On current system (PowerShell):
tar --exclude='.venv' --exclude='.pytest_cache' --exclude='__pycache__' -czvf allmimic-nao.tar.gz .
```
Copy `allmimic-nao.tar.gz` to your A4000 machine and extract:
```bash
tar -xzvf allmimic-nao.tar.gz -C allmimic-nao
cd allmimic-nao
```

---

## 4. Setting up Environment on RTX A4000 (Ubuntu / Linux)

```bash
# 1. Create Python 3.10 or 3.11/3.12 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA 12 support
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install mjlab and mink
pip install mink mediapy pytest
```

### Run Unit Tests on the A4000 machine:
```bash
pytest tests/test_nao_fk.py tests/test_nao_task.py tests/test_motion_lib_nao.py -v
```
All 13 tests should pass in under 3 seconds.

---

## 5. Training Execution Protocol on RTX A4000 (16GB VRAM)

### Stage A: Single-Clip Smoke Test (~15 mins)
Verify reward convergence and World Model loss stability on a single walking clip:

```bash
python -m allmimic.train \
  --preset recommended \
  --motions data/processed/nao_lafan_subset_50hz.npz \
  --clip-prefixes walk_forward \
  --num-envs 2048 \
  --horizon 32 \
  --iterations 3000 \
  --out runs/nao_smoke_stageA
```

**Expected Criteria:**
- **VRAM**: ~5.5 GB peak.
- **Speed**: $< 0.18\text{ s/iteration}$ with CUDA graphs.
- **Normalized return**: $> 0.85$ within 2,500 iterations.
- If normalized return plateaus $< 0.3$, verify action scales (`0.25` legs, `0.35` arms in `nao_config.py`).

---

### Stage B: 12-Clip Multi-Skill Library Training (~1.5–2 hours)
Train the policy on the complete 12-clip motion library:

```bash
python -m allmimic.train \
  --preset recommended \
  --motions data/processed/nao_lafan_subset_50hz.npz \
  --num-envs 4096 \
  --horizon 32 \
  --iterations 25000 \
  --eval-every 2500 \
  --out runs/nao_12clips_stageB
```

**Expected Criteria:**
- **VRAM**: ~10.5 GB (fits comfortably in RTX A4000 16GB).
- **Speed**: ~0.24 s/iter (~1.7 hours wall-clock for 25,000 iters).
- **Evaluation target**: $12/12$ zero-fall completions, normalized return $> 0.82$.

---

## 6. Evaluation and Verification

To benchmark a trained checkpoint across all 12 clips:
```bash
python -m allmimic.evaluate \
  --run runs/nao_12clips_stageB \
  --motions data/processed/nao_lafan_subset_50hz.npz \
  --num-envs 12
```

To re-render videos of the dataset or rollouts at any time:
```bash
python scripts/visualize_clips.py
```

---

## 7. Key Files Manifest

- `assets/nao/nao_supermimic.xml`: 24-DoF stripped NAO MJCF model with 10 tracking sites.
- `allmimic/nao_config.py`: `mjlab` Entity configuration, actuator gains, and initial balance posture.
- `allmimic/kinematics.py`: Rodrigues PyTorch batched differentiable forward kinematics.
- `allmimic/task.py`: 24-DoF / 23-Action dimension definitions, BeyondMimic scaled reward kernels.
- `allmimic/joint_weights.py`: Holden joint error propagation weights for NAO.
- `allmimic/train.py`: SuperMimic training loop with World Model & policy BPTT.
- `scripts/retarget_lafan_to_nao.py`: `mink` QP retargeter & dataset generator.
- `scripts/visualize_clips.py`: Offscreen MuJoCo renderer for MP4 video generation.
- `videos/*.mp4`: 12 rendered MP4 reference motion videos.
- `tests/test_nao_*.py`: Full unit test suite covering FK parity, task dimensions, and motion lib.

