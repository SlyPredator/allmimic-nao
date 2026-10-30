# Research Stick-Figure Humanoid Representation

This folder contains a **kinematically verified stick-figure humanoid** and an interactive viewer to animate reference motions across **LAFAN1** and **BONES-SEED** motion datasets.

---

## 1. Datasets & Motion Catalogs

Detailed manifests listing every clip, its CLI category keyword, frame count, and duration are in [`lists/`](lists/):

### A. LAFAN1 Mocap Clips (40 Clips)
Stored under `reference_motions/data_lafan1/g1/`:
- **Catalog**: [`lists/lafan1_motions.csv`](lists/lafan1_motions.csv) (or plain names in [`lists/lafan1_motions.txt`](lists/lafan1_motions.txt))
- **Category Keywords**: `dance`, `fallAndGetUp`, `fight`, `fightAndSports`, `jumps`, `run`, `sprint`, `walk`

### B. BONES-SEED Dataset (1,388+ Clips)
Stored under `reference_motions/data_bones_seed/g1/`:
- **Catalog**: [`lists/bones_seed_motions.csv`](lists/bones_seed_motions.csv) (or plain names in [`lists/bones_seed_motions.txt`](lists/bones_seed_motions.txt))
- **Category Keywords**: `shadow_boxing`, `ab_bicycle`, `vertical_mountain_climber`, `box_jump`, `burpee`, `warm_up_legs`, `warm_up_arms`, `elbow_plank`, `forward_lunge`, `jumping_rope`, `push_up`, etc.

> **Tip**: Any value in the `category` column can be passed directly to `--clip <category>` in the viewer.

---

## 2. Setup & Download Scripts

To set up or restore the motion datasets locally:

```bash
# Download / restore all 40 LAFAN1 clips into reference_motions/data_lafan1/g1/
python reference_motions/scripts/download_lafan1.py

# Download & stream-extract BONES-SEED clips from Hugging Face into reference_motions/data_bones_seed/g1/
python reference_motions/scripts/download_bones_seed.py
```
*(Note: BONES-SEED requires that you have accepted dataset permissions on [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) with your Hugging Face token).*

---

## 3. Interactive 3D Viewers

### LAFAN1 Motions:
```bash
# Launch interactive 40-clip viewer:
python reference_motions/animate_lafan1.py

# Jump directly to a clip by name or index:
python reference_motions/animate_lafan1.py --clip fallAndGetUp1_subject1
python reference_motions/animate_lafan1.py --clip run1_subject2
python reference_motions/animate_lafan1.py --clip 8
```

### BONES-SEED Motions:
```bash
# Launch interactive viewer across all 1,388 BONES-SEED clips:
python reference_motions/animate_bones_seed.py

# Jump directly to specific BONES-SEED motions by name:
python reference_motions/animate_bones_seed.py --clip shadow_boxing
python reference_motions/animate_bones_seed.py --clip vertical_mountain_climber
python reference_motions/animate_bones_seed.py --clip warm_up_legs
```

---

## 4. In-Viewer Controls

| Key | Action |
| :--- | :--- |
| **`]` / `[`** | **Next / Previous clip** (cycle through all loaded clips) |
| **`Space`** | **Pause / Resume** playback |
| **`R`** | **Reset** playback to Frame 0 |
| **`->` / `<-`** | **Step forward / backward 15 frames** (when paused or scrubbing) |
| **`^` / `v`** | **Speed up / Slow down** playback rate ($0.25\times$ to $3.0\times$) |
| **Mouse Drag** | Orbit (left-click), Pan (right-click), Zoom (scroll wheel) |
