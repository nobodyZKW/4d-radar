from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@torch.no_grad()
def run_eval_once(cfg_obj, model, dataloader):
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    model.eval()
    for batch_dict in dataloader:
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model(batch_dict)
        annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None)
        det_annos.extend(annos)

    _, metrics = dataset.evaluation(
        det_annos,
        class_names,
        eval_metric=cfg_obj.MODEL.POST_PROCESSING.EVAL_METRIC,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser("Decode/NMS tuning for OpenPCDet CenterHead")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--score-thresh", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--nms-thresh", type=str, default="0.30,0.50,0.70")
    parser.add_argument("--post-maxsize", type=str, default="100,300,500")
    parser.add_argument("--nms-types", type=str, default="class_agnostic_nms,class_specific_nms")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()

    _, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_loader.dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()

    score_grid = parse_float_list(args.score_thresh)
    nms_grid = parse_float_list(args.nms_thresh)
    post_grid = parse_int_list(args.post_maxsize)
    nms_types = [x.strip() for x in args.nms_types.split(",") if x.strip()]

    rows = []
    best = None

    for nms_type, score_t, nms_t, post_k in itertools.product(nms_types, score_grid, nms_grid, post_grid):
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = float(score_t)
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE = int(post_k)
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_TYPE = nms_type

        if nms_type == "class_specific_nms":
            n_cls = len(cfg.CLASS_NAMES)
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = [float(nms_t)] * n_cls
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE = [4096] * n_cls
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = [int(post_k)] * n_cls
        else:
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = float(nms_t)
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE = 4096
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE = int(post_k)

        if nms_type == "circle_nms":
            # Current OpenPCDet release keeps circle_nms path as not-ready in CenterHead.
            row = {
                "nms_type": nms_type,
                "score_thresh": float(score_t),
                "nms_thresh": float(nms_t),
                "post_maxsize": int(post_k),
                "quick/mean_f1": -1.0,
                "note": "circle_nms unavailable in current OpenPCDet CenterHead",
            }
            rows.append(row)
            continue

        metrics = run_eval_once(cfg, model, test_loader)
        row = {
            "nms_type": nms_type,
            "score_thresh": float(score_t),
            "nms_thresh": float(nms_t),
            "post_maxsize": int(post_k),
            "quick/mean_f1": float(metrics.get("quick/mean_f1", 0.0)),
            "quick/Car_f1": float(metrics.get("quick/Car_f1", 0.0)),
            "quick/Pedestrian_f1": float(metrics.get("quick/Pedestrian_f1", 0.0)),
            "quick/Cyclist_f1": float(metrics.get("quick/Cyclist_f1", 0.0)),
        }
        rows.append(row)

        if best is None or row["quick/mean_f1"] > best["quick/mean_f1"]:
            best = row

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"rows": rows, "best": best}, ensure_ascii=False, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved tuning rows: {output_json}")
    print(f"Saved tuning table: {output_csv}")
    if best is not None:
        print("Best setting:")
        print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
