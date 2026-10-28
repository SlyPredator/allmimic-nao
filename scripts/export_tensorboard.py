"""Export JSONL training & evaluation logs to TensorBoard format.

Usage:
    conda run -n supermimic_nao python scripts/export_tensorboard.py --run runs/nao_smoke_stageA
"""

import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter


def export_run(run_dir: str):
    log_file = os.path.join(run_dir, "log.jsonl")
    eval_file = os.path.join(run_dir, "eval.jsonl")

    if not os.path.isfile(log_file):
        print(f"Error: No log.jsonl found in {run_dir}")
        return

    writer = SummaryWriter(log_dir=run_dir)
    print(f"Exporting logs from {run_dir} to TensorBoard...")

    # 1. Training metrics
    count = 0
    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            it = rec.get("it", count * 100)
            count += 1
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    if k.startswith("img_r_"):
                        tag = f"rewards/{k[6:]}"
                    elif k.startswith("wm_"):
                        tag = f"world_model/{k[3:]}"
                    elif k.startswith("pi_"):
                        tag = f"policy/{k[3:]}"
                    elif k.startswith("ep_"):
                        tag = f"train/{k[3:]}"
                    elif k in ("img_survival", "hours", "iter_s", "lr", "v_loss", "v_mean", "mem_gb"):
                        tag = f"train/{k}"
                    else:
                        tag = f"stats/{k}"
                    writer.add_scalar(tag, v, it)

    # 2. Evaluation metrics
    if os.path.isfile(eval_file):
        with open(eval_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                res = json.loads(line)
                it = res.get("it", 0)
                if "norm_return" in res:
                    writer.add_scalar("eval/norm_return", res["norm_return"], it)
                if "norm_return_frame_weighted" in res:
                    writer.add_scalar("eval/norm_return_frame_weighted", res["norm_return_frame_weighted"], it)
                if "failed" in res:
                    writer.add_scalar("eval/failed", res["failed"], it)
                if "num_clips" in res:
                    writer.add_scalar("eval/num_clips", res["num_clips"], it)
                if "per_clip_norm_return" in res:
                    for i, score in enumerate(res["per_clip_norm_return"]):
                        writer.add_scalar(f"eval_clips/clip_{i}", score, it)

    writer.flush()
    writer.close()
    print(f"Successfully converted {count} log checkpoints to TensorBoard in: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert jsonl run logs to TensorBoard")
    parser.add_argument("--run", default="runs/nao_smoke_stageA", help="Run directory to convert")
    args = parser.parse_args()
    export_run(args.run)


if __name__ == "__main__":
    main()
