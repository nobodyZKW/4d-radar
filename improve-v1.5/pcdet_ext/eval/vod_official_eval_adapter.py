from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

from .export_kitti_like_predictions import export_kitti_like_predictions


def main():
    parser = argparse.ArgumentParser("VoD official evaluator adapter")
    parser.add_argument("--result-pkl", type=str, required=True, help="Path to OpenPCDet result.pkl")
    parser.add_argument("--output-dir", type=str, required=True, help="Export dir for per-frame txt")
    parser.add_argument(
        "--devkit-cmd",
        type=str,
        default="",
        help="Optional external devkit command. Use {pred_dir} placeholder for exported prediction folder.",
    )
    args = parser.parse_args()

    result_pkl = Path(args.result_pkl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(result_pkl, "rb") as f:
        det_annos = pickle.load(f)

    pred_dir = output_dir / "predictions"
    export_kitti_like_predictions(det_annos=det_annos, output_dir=pred_dir)
    print(f"Exported prediction txt files to: {pred_dir}")

    if args.devkit_cmd:
        cmd = args.devkit_cmd.format(pred_dir=str(pred_dir))
        print(f"Running devkit command: {cmd}")
        completed = subprocess.run(cmd, shell=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Devkit command failed with code {completed.returncode}")


if __name__ == "__main__":
    main()
