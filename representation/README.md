# Research Stick-Figure Humanoid Representation

This folder contains a **kinematically verified stick-figure humanoid** and an interactive viewer to animate through all **40 original LAFAN1 motion-capture clips**.

---

## 1. All 40 LAFAN1 Mocap Clips

All 40 raw clips are stored under `representation/data_lafan1/g1/`:

| Index | Category | Clips |
| :---: | :--- | :--- |
| **0 – 7** | **Dance** | `dance1_subject1..3`, `dance2_subject1..5` |
| **8 – 13** | **Fall & Get Up** | `fallAndGetUp1_subject1,4,5`, `fallAndGetUp2_subject2,3`, `fallAndGetUp3_subject1` |
| **14 – 18** | **Fight & Sports** | `fight1_subject2,3,5`, `fightAndSports1_subject1,4` |
| **19 – 21** | **Jumps** | `jumps1_subject1,2,5` |
| **22 – 27** | **Run & Sprint** | `run1_subject2,5`, `run2_subject1,4`, `sprint1_subject2,4` |
| **28 – 39** | **Walks** | `walk1_subject1,2,5`, `walk2_subject1,3,4`, `walk3_subject1..5`, `walk4_subject1` |

---

## 2. Interactive 3D Viewer

Launch the animator:

```bash
python representation/animate_lafan1.py
```

### Jump to specific clips directly:
```bash
# By name:
python representation/animate_lafan1.py --clip fallAndGetUp1_subject1
python representation/animate_lafan1.py --clip run1_subject2
python representation/animate_lafan1.py --clip dance2_subject1

# By index (0 - 39):
python representation/animate_lafan1.py --clip 8
```

---

## 3. In-Viewer Controls

| Key | Action |
| :--- | :--- |
| **`]` / `[`** | **Next / Previous clip** (cycle seamlessly through all 40 clips) |
| **`Space`** | **Pause / Resume** playback |
| **`R`** | **Reset** playback to Frame 0 |
| **`->` / `<-`** | **Step forward / backward 15 frames** (when paused or scrubbing) |
| **`^` / `v`** | **Speed up / Slow down** playback rate ($0.25\times$ to $3.0\times$) |
| **Mouse Drag** | Orbit (left-click), Pan (right-click), Zoom (scroll wheel) |
