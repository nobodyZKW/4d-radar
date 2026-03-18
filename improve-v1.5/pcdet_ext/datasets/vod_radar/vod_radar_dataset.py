from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.kitti import kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import calibration_kitti, common_utils

try:
    from .vod_radar_utils import (
        DEFAULT_CLASS_MAP,
        get_image_shape,
        load_vod_calib,
        map_class_name,
        parse_vod_label_file,
        resolve_image_file,
    )
except ImportError:
    from vod_radar_utils import (
        DEFAULT_CLASS_MAP,
        get_image_shape,
        load_vod_calib,
        map_class_name,
        parse_vod_label_file,
        resolve_image_file,
    )

try:
    from ...utils.velocity_supervision import VelocitySupervisionEstimator
except ImportError:
    from pcdet_ext.utils.velocity_supervision import VelocitySupervisionEstimator


class VodRadarDataset(DatasetTemplate):
    """VoD 4D radar dataset in OpenPCDet style.

    Expected directory layout under DATA_PATH:
      - radar_5frames/training/velodyne/*.bin
      - lidar/training/label_2/*.txt
      - lidar/training/calib/*.txt
      - lidar/training/image_2/*.{jpg,png}
      - lidar/ImageSets/{train,val,test}.txt
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.class_map = self.dataset_cfg.get("CLASS_MAP", DEFAULT_CLASS_MAP)
        self.map_class_to_kitti = self.dataset_cfg.get(
            "MAP_CLASS_TO_KITTI", {name: name for name in self.class_names}
        )

        self.point_dim = int(self.dataset_cfg.get("POINT_DIM", len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)))
        self.feature_indices = self.dataset_cfg.get(
            "RADAR_FEATURE_INDEX", {"rcs": 3, "v_r": 4, "v_r_comp": 5, "time": 6}
        )

        self.fov_points_only = bool(self.dataset_cfg.get("FOV_POINTS_ONLY", False))
        self.append_velocity = bool(self.dataset_cfg.get("APPEND_VELOCITY_TO_GT_BOXES", False))
        self.point_dropout_prob = float(self.dataset_cfg.get("POINT_DROPOUT_PROB", 0.0))

        vel_cfg = self.dataset_cfg.get("VELOCITY_SUPERVISION", {})
        self.velocity_estimator = VelocitySupervisionEstimator(vel_cfg) if self.append_velocity else None

        self.paths = self._resolve_paths()
        self.sample_id_list = self._load_split_ids(self.split)

        self.vod_infos: List[Dict] = []
        self.include_vod_radar_data(self.mode)

    def _resolve_paths(self) -> Dict[str, Path]:
        base = Path(self.root_path)
        return {
            "radar": base / "radar_5frames" / "training" / "velodyne",
            "labels": base / "lidar" / "training" / "label_2",
            "calib": base / "lidar" / "training" / "calib",
            "images": base / "lidar" / "training" / "image_2",
            "imagesets": base / "lidar" / "ImageSets",
        }

    def _load_split_ids(self, split: str) -> List[str]:
        split_file = self.paths["imagesets"] / f"{split}.txt"
        if not split_file.exists():
            return []
        return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]

    def include_vod_radar_data(self, mode: str) -> None:
        if self.logger is not None:
            self.logger.info("Loading VodRadar dataset")

        infos: List[Dict] = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            p = Path(self.root_path) / info_path
            if not p.exists():
                continue
            with open(p, "rb") as f:
                infos.extend(pickle.load(f))

        self.vod_infos.extend(infos)
        if self.logger is not None:
            self.logger.info("Total samples for VodRadar: %d", len(self.vod_infos))

    def set_split(self, split: str) -> None:
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split
        self.sample_id_list = self._load_split_ids(split)

    def get_lidar(self, idx: str) -> np.ndarray:
        bin_path = self.paths["radar"] / f"{idx}.bin"
        points = np.fromfile(str(bin_path), dtype=np.float32)
        if points.size % self.point_dim != 0:
            raise ValueError(f"Invalid radar point dim: {bin_path}, size={points.size}, dim={self.point_dim}")
        return points.reshape(-1, self.point_dim)

    def get_calib(self, idx: str) -> calibration_kitti.Calibration:
        calib_file = self.paths["calib"] / f"{idx}.txt"
        return calibration_kitti.Calibration(calib_file)

    def get_image_shape(self, idx: str) -> np.ndarray:
        image_file = resolve_image_file(self.paths["images"], idx)
        if image_file is None:
            return np.array([0, 0], dtype=np.int32)
        return get_image_shape(image_file)

    @staticmethod
    def get_fov_flag(pts_rect: np.ndarray, img_shape: np.ndarray, calib: calibration_kitti.Calibration) -> np.ndarray:
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        return np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    def get_infos(
        self,
        class_names: List[str],
        num_workers: int = 4,
        has_label: bool = True,
        count_inside_pts: bool = True,
        sample_id_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        import concurrent.futures as futures

        ids = sample_id_list if sample_id_list is not None else self.sample_id_list

        def process_one(sample_idx: str) -> Dict:
            info: Dict = {}
            info["point_cloud"] = {
                "num_features": self.point_dim,
                "lidar_idx": sample_idx,
            }

            image_shape = self.get_image_shape(sample_idx)
            info["image"] = {"image_idx": sample_idx, "image_shape": image_shape}

            calib_path = self.paths["calib"] / f"{sample_idx}.txt"
            info["calib"] = load_vod_calib(calib_path)

            if has_label:
                label_path = self.paths["labels"] / f"{sample_idx}.txt"
                annos = parse_vod_label_file(
                    label_path=label_path,
                    tr_velo_to_cam=info["calib"]["Tr_velo_to_cam"],
                    class_map=self.class_map,
                    class_names=class_names,
                )

                if count_inside_pts and annos["gt_boxes_lidar"].shape[0] > 0:
                    points = self.get_lidar(sample_idx)
                    num_gt = annos["gt_boxes_lidar"].shape[0]
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(points[:, :3]), torch.from_numpy(annos["gt_boxes_lidar"])
                    ).numpy()
                    num_points_in_gt = np.zeros((num_gt,), dtype=np.int32)
                    for i in range(num_gt):
                        num_points_in_gt[i] = int((point_indices[i] > 0).sum())
                    annos["num_points_in_gt"] = num_points_in_gt

                info["annos"] = annos

            return info

        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            infos = executor.map(process_one, ids)
        return list(infos)

    def create_groundtruth_database(self, info_path: Path, used_classes=None, split="train"):
        db_save_path = Path(self.root_path) / ("gt_database" if split == "train" else f"gt_database_{split}")
        db_info_save_path = Path(self.root_path) / f"vod_radar_dbinfos_{split}.pkl"
        db_save_path.mkdir(parents=True, exist_ok=True)

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        all_db_infos = {}
        for k, info in enumerate(infos):
            sample_idx = info["point_cloud"]["lidar_idx"]
            points = self.get_lidar(sample_idx)
            annos = info["annos"]
            names = annos["name"]
            gt_boxes = annos["gt_boxes_lidar"]
            if gt_boxes.shape[0] == 0:
                continue

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(gt_boxes.shape[0]):
                filename = f"{sample_idx}_{names[i]}_{i}.bin"
                filepath = db_save_path / filename
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "wb") as f_obj:
                    gt_points.tofile(f_obj)

                if used_classes is None or names[i] in used_classes:
                    db_info = {
                        "name": names[i],
                        "path": str(filepath.relative_to(self.root_path)),
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                    }
                    all_db_infos.setdefault(names[i], []).append(db_info)

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.vod_infos) * self.total_epochs
        return len(self.vod_infos)

    def _estimate_velocity_targets(
        self, points: np.ndarray, gt_boxes_lidar: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.velocity_estimator is None or gt_boxes_lidar.shape[0] == 0:
            vel = np.full((gt_boxes_lidar.shape[0], 3), np.nan, dtype=np.float32)
            return vel, {
                "num_valid_vel_boxes": 0.0,
                "num_weak_vel_boxes": 0.0,
                "vel_fit_residual_mean": 0.0,
            }

        vel_targets, _, stats = self.velocity_estimator.estimate_for_boxes(
            points=points,
            gt_boxes_lidar=gt_boxes_lidar,
            feature_indices=self.feature_indices,
        )
        return vel_targets, stats

    def __getitem__(self, index: int):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.vod_infos)

        info = copy.deepcopy(self.vod_infos[index])
        sample_idx = info["point_cloud"]["lidar_idx"]
        points = self.get_lidar(sample_idx)
        if self.training and self.point_dropout_prob > 0 and points.shape[0] > 0:
            keep_mask = np.random.rand(points.shape[0]) > self.point_dropout_prob
            if keep_mask.any():
                points = points[keep_mask]

        input_dict = {
            "frame_id": sample_idx,
            "points": points,
        }

        if "annos" in info:
            annos = common_utils.drop_info_with_name(info["annos"], name="DontCare")
            gt_names = annos["name"]
            gt_boxes_lidar = annos["gt_boxes_lidar"].astype(np.float32)

            if self.append_velocity:
                vel_targets, vel_stats = self._estimate_velocity_targets(points, gt_boxes_lidar)
                gt_boxes_lidar = np.concatenate([gt_boxes_lidar, vel_targets], axis=1)
                input_dict.update(vel_stats)

            input_dict.update(
                {
                    "gt_names": gt_names,
                    "gt_boxes": gt_boxes_lidar,
                }
            )

        if self.fov_points_only:
            calib = self.get_calib(sample_idx)
            image_shape = info.get("image", {}).get("image_shape", self.get_image_shape(sample_idx))
            pts_rect = calib.lidar_to_rect(points[:, :3])
            fov_mask = self.get_fov_flag(pts_rect, image_shape, calib)
            input_dict["points"] = points[fov_mask]

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if len(self.vod_infos) == 0 or "annos" not in self.vod_infos[0]:
            return "No ground-truth boxes for evaluation", {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.vod_infos]

        # KITTI-style official metrics.
        kitti_class_names = [self.map_class_to_kitti[x] for x in class_names]
        kitti_utils.transform_annotations_to_kitti_format(
            eval_det_annos, map_name_to_kitti=self.map_class_to_kitti
        )
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos,
            map_name_to_kitti=self.map_class_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False),
        )

        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos,
            dt_annos=eval_det_annos,
            current_classes=kitti_class_names,
        )

        quick = self._quick_center_f1(det_annos)
        ap_dict.update(quick)
        result_str = ap_result_str + "\n" + "\n".join([f"{k}: {v:.6f}" for k, v in quick.items()])
        return result_str, ap_dict

    def _quick_center_f1(self, det_annos: List[Dict]) -> Dict[str, float]:
        thresh = float(self.dataset_cfg.get("QUICK_EVAL_CENTER_DIST", 2.0))
        cls_to_idx = {name: i for i, name in enumerate(self.class_names)}
        stats = [{"tp": 0, "fp": 0, "gt": 0} for _ in self.class_names]

        for gt_info, dt in zip(self.vod_infos, det_annos):
            gt_ann = gt_info.get("annos", {})
            gt_names = gt_ann.get("name", np.zeros((0,), dtype=object))
            gt_boxes = gt_ann.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32))
            dt_names = dt.get("name", np.zeros((0,), dtype=object))
            dt_boxes = dt.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))
            dt_scores = dt.get("score", np.zeros((0,), dtype=np.float32))

            used = np.zeros((len(gt_names),), dtype=bool)
            order = np.argsort(-dt_scores)
            for oi in order:
                name = str(dt_names[oi])
                if name not in cls_to_idx:
                    continue
                cls_id = cls_to_idx[name]
                box = dt_boxes[oi]

                best_i = -1
                best_d = 1e9
                for gi, (gname, gbox) in enumerate(zip(gt_names, gt_boxes)):
                    if used[gi] or str(gname) != name:
                        continue
                    d = float(np.linalg.norm(box[:2] - gbox[:2]))
                    if d < best_d:
                        best_d = d
                        best_i = gi

                if best_i >= 0 and best_d <= thresh:
                    used[best_i] = True
                    stats[cls_id]["tp"] += 1
                else:
                    stats[cls_id]["fp"] += 1

            for gname in gt_names:
                if str(gname) in cls_to_idx:
                    stats[cls_to_idx[str(gname)]]["gt"] += 1

        out = {}
        mean_f1 = 0.0
        for cls_name, cls_id in cls_to_idx.items():
            tp = stats[cls_id]["tp"]
            fp = stats[cls_id]["fp"]
            gt = stats[cls_id]["gt"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(gt, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            out[f"quick/{cls_name}_precision"] = float(prec)
            out[f"quick/{cls_name}_recall"] = float(rec)
            out[f"quick/{cls_name}_f1"] = float(f1)
            mean_f1 += f1

        out["quick/mean_f1"] = float(mean_f1 / max(len(cls_to_idx), 1))
        return out


def create_vod_radar_infos(dataset_cfg, class_names, data_path: Path, save_path: Path, workers: int = 4):
    dataset = VodRadarDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=False,
        logger=common_utils.create_logger(),
    )

    train_split = dataset_cfg.DATA_SPLIT.get("train", "train")
    val_split = dataset_cfg.DATA_SPLIT.get("test", "val")
    test_split = dataset_cfg.DATA_SPLIT.get("test", "test")

    train_file = save_path / "vod_radar_infos_train.pkl"
    val_file = save_path / "vod_radar_infos_val.pkl"
    trainval_file = save_path / "vod_radar_infos_trainval.pkl"
    test_file = save_path / "vod_radar_infos_test.pkl"

    print("--------------- Start generating VoD radar infos ---------------")

    dataset.set_split(train_split)
    infos_train = dataset.get_infos(class_names=class_names, num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_file, "wb") as f:
        pickle.dump(infos_train, f)
    print(f"Saved: {train_file}")

    dataset.set_split(val_split)
    infos_val = dataset.get_infos(class_names=class_names, num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_file, "wb") as f:
        pickle.dump(infos_val, f)
    print(f"Saved: {val_file}")

    with open(trainval_file, "wb") as f:
        pickle.dump(infos_train + infos_val, f)
    print(f"Saved: {trainval_file}")

    test_has_label = bool(dataset_cfg.get("TEST_HAS_LABEL", False))
    dataset.set_split(test_split)
    infos_test = dataset.get_infos(
        class_names=class_names,
        num_workers=workers,
        has_label=test_has_label,
        count_inside_pts=test_has_label,
    )
    with open(test_file, "wb") as f:
        pickle.dump(infos_test, f)
    print(f"Saved: {test_file}")

    if dataset_cfg.get("CREATE_GT_DATABASE", False):
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(info_path=train_file, split=train_split)
        print("Ground truth database created.")

    print("--------------- VoD radar info generation done ---------------")


if __name__ == "__main__":
    import sys
    import yaml
    from easydict import EasyDict

    if len(sys.argv) >= 3 and sys.argv[1] == "create_vod_radar_infos":
        cfg = EasyDict(yaml.safe_load(open(sys.argv[2], "r", encoding="utf-8")))
        if "DATA_CONFIG" in cfg:
            dataset_cfg = cfg.DATA_CONFIG
            class_names = cfg.CLASS_NAMES
        else:
            dataset_cfg = cfg
            class_names = dataset_cfg.CLASS_NAMES
        data_root = Path(dataset_cfg.DATA_PATH)
        create_vod_radar_infos(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            data_path=data_root,
            save_path=data_root,
            workers=4,
        )
