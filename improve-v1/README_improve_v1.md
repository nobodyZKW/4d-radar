# improve-v1: CenterPoint 化 + Radar-specific Enhancement

本目录是对 `baseline/` 的**增量式 v1 改进**，目标是可运行、可验证、可对比。

## 1. 改进目标

- 保留 center-based 单阶段框架
- 显式增强雷达特征利用（RCS / doppler / compensated_doppler / time）
- 加入更规范的检测头与后处理
- 保持原 `baseline/` 不受影响

## 2. 文件结构

- `improve-v1/dataset.py`
  - 16 通道 BEV 特征
  - 数据增强（flip / scale / rotation / point dropout）
  - 新 target 组织（heatmap / offset / z / size / yaw / vel）
- `improve-v1/model.py`
  - 轻量双分支（geometry + motion）
  - center-based 多头（含 velocity head）
- `improve-v1/train.py`
  - 分项 loss
  - decode（local max + topk + center/circle NMS）
  - scheduler + EMA
- `improve-v1/eval.py`
  - 独立评估入口（支持 `--use-ema`）
- `improve-v1/configs/vod_centerpoint_radar_v1.yaml`
  - v1 配置
- `improve-v1/scripts/train_v1.ps1`
- `improve-v1/scripts/eval_v1.ps1`

## 3. v1 模型与头部

输出头：

1. `heatmap`
2. `offset` (`dx, dy`)
3. `z`
4. `size` (`l, w, h` in log space)
5. `yaw` (`sin(yaw), cos(yaw)`)
6. `vel` (`vx, vy`) 头**始终保留**

说明：
- `velocity head` 永远存在；
- `vel loss` 由 `train.use_velocity_loss` 控制；
- 若没有稳定速度真值，可先关闭 `vel loss`。

## 4. Radar-specific BEV 特征（extended16）

通道顺序：

1. `count(log1p)`
2. `z_mean`
3. `z_max`
4. `z_std`
5. `rcs_mean`
6. `rcs_max`
7. `rcs_std`
8. `doppler_mean`
9. `doppler_min`
10. `doppler_max`
11. `comp_doppler_mean`
12. `comp_doppler_min`
13. `comp_doppler_max`
14. `time_mean`
15. `time_max`
16. `time_std`

空 cell 采用安全填充（0），并做数值稳定处理（方差下限截断）。

## 5. Decode / 后处理

- heatmap 局部峰值抑制（`max_pool2d(3,1,1)`）
- top-k 解码（`per_class` 或 `global`）
- NMS（`circle/center` 基于中心距离）

关键配置：

- `decode.score_thresh`
- `decode.topk`
- `decode.topk_mode`
- `decode.nms_type`
- `decode.nms_thresh`

## 6. Loss 设计

- `hm loss`：focal
- `box loss`：`offset + z + size`（smooth L1）
- `yaw loss`：smooth L1
- `vel loss`：smooth L1（可开关）

权重：

- `train.hm_weight`
- `train.box_weight`
- `train.yaw_weight`
- `train.vel_weight`

日志包含：`total/hm/box/yaw/vel/lr`。

## 7. 训练策略

- `scheduler`：`none / cosine / multistep`
- `EMA`：可开关，支持 `eval_with_ema`

## 8. 运行方式

### 安装依赖

```powershell
cd E:\毕设\code\4d-radar
pip install -r .\improve-v1\requirements.txt
```

### 训练

```powershell
cd E:\毕设\code\4d-radar
python .\improve-v1\train.py --config .\improve-v1\configs\vod_centerpoint_radar_v1.yaml
```

或

```powershell
.\improve-v1\scripts\train_v1.ps1
```

### 验证

```powershell
cd E:\毕设\code\4d-radar
python .\improve-v1\eval.py `
  --config .\improve-v1\configs\vod_centerpoint_radar_v1.yaml `
  --ckpt .\improve-v1\outputs\vod_centerpoint_radar_v1\best.pt `
  --use-ema
```

或

```powershell
.\improve-v1\scripts\eval_v1.ps1
```

## 9. 对比实验切换建议

仅通过 config 切换即可：

1. baseline-like:
   - `configs/vod_ablation_baseline_like.yaml`
2. baseline + 后处理增强:
   - `configs/vod_ablation_baseline_post.yaml`
3. baseline + 特征增强:
   - `configs/vod_ablation_feature_enhanced.yaml`
4. centerpoint_radar_v1 全量:
   - `configs/vod_centerpoint_radar_v1.yaml`

## 10. 已知限制（v1）

- 当前公开标注不直接给 `vx, vy`，默认 `vel loss` 关闭；
- `velocity_target_mode=comp_doppler_heading` 是近似方案，仅用于接口联调，不是最终物理最优方案；
- 评估仍采用中心点匹配 F1（非官方 3D mAP）。
