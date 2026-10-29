# SuperMimic Web Deployment (`web_dep`)

An interactive, client-side 3D web visualizer for SuperMimic whole-body humanoid tracking policies (NAO / G1 / Mechlab Biped), styled after the official SuperMimic studio demo.

---

## Features

- **Three Visual Layers:**
  - 🟢 **MuJoCo:** Real-time physics / kinematic state with user-adjustable noise and friction.
  - ⚪ **Goal Ghost:** The semi-transparent white reference motion target (*"the pose it's chasing"*).
  - 🔵 **World Model Ghost:** The semi-transparent blue predictive imagination phantom.
- **Interactive Controls:**
  - Timeline scrubbing and play/pause controls.
  - Motion clip selector dropdown.
  - Exploration action noise slider (`0.00` to `0.50`).
  - Contact friction slider (`10%` to `150%`).
  - Camera orbit, pan, and smooth zoom with auto-tracking.
- **Zero Backend Required:** Runs 100% in the user's browser using WebGL and WebAssembly (can be hosted for free on GitHub Pages, Vercel, or Netlify).

---

## Directory Structure

```
web_dep/
├── scripts/
│   └── export_web_assets.py    # Exports policy to ONNX, motions to JSON, and parity test vectors
├── public/
│   ├── index.html              # Modern SuperMimic studio layout & controls
│   ├── css/
│   │   └── style.css           # Glassmorphic Apple-style studio styling
│   ├── js/
│   │   ├── app.js              # Three.js 3D scene, kinematic rig, and timeline controller
│   │   └── worker.js           # Web Worker interface for physics & policy evaluation
│   ├── assets/                 # Robot XML and visual mesh assets
│   ├── config.json             # Joint names, action scales, and normalizer statistics
│   ├── motions.json            # Reference motion clip trajectories
│   ├── parity_test.json        # Test vector ensuring JS/ONNX math matches PyTorch exactly
│   └── policy.onnx             # Exported PyTorch actor policy
├── serve.py                    # Local dev server with COOP/COEP headers
└── README.md                   # Documentation
```

---

## Quickstart

### 1. Export Model & Motions (Already Done for `nao_walk_forward`)
To re-export or update from any checkpoint:

```bash
conda activate supermimic_nao
python web_dep/scripts/export_web_assets.py \
    --run runs/nao_walk_forward \
    --out-dir web_dep/public \
    --num-clips 5
```

### 2. Launch Local Visualizer
Run the included dev server:

```bash
python web_dep/serve.py --port 8080
```

Open [http://localhost:8080](http://localhost:8080) in your web browser.

---

## Keyboard Shortcuts

- `[SPACE]` : Play / Pause
- `[R]` : Reset current clip to start
- `[G]` : Toggle Goal ghost (White)
- `[W]` : Toggle World Model ghost (Blue)
