# improve-v1.5：OpenPCDet 版 4D 雷达 3D 检测

本目录实现了一个基于 OpenPCDet 的 `v1.5` 工程，主线为：

- `VodRadarDataset`（VoD 4D radar 5-frames）
- `Radar7PillarVFE`（7维雷达特征 + 几何/运动双流）
- `CenterPoint + CenterHead`
- 速度监督（鲁棒 2D 径向约束 + 弱监督回退）
- decode/NMS 可配置与调参脚本
- A0~A4 + 特征开关 + 速度监督质量 消融脚本

## 1. 目录结构

- `external/OpenPCDet`: 原始 OpenPCDet 代码（已做最小注册/接口补丁）
- `pcdet_ext/datasets/vod_radar`: 数据集与 info 生成
- `pcdet_ext/models/backbones_3d/vfe/radar7_pillar_vfe.py`: 自定义 VFE
- `pcdet_ext/utils/velocity_supervision.py`: 速度监督求解器
- `pcdet_ext/utils/decode_tuning.py`: decode/NMS 网格搜索
- `pcdet_ext/utils/ablation_runner.py`: 消融批量运行与汇总
- `pcdet_ext/eval/*`: 预测导出与官方评估适配
- `configs/dataset_configs/vod_radar_dataset.yaml`: 数据集配置
- `configs/model_configs/*.yaml`: v1.5 与消融配置
- `scripts/*.ps1`: 数据准备、训练、评估、消融、导出脚本

## 2. 环境安装

```powershell
cd E:\毕设\code\4d-radar\improve-v1.5
pip install -r requirements.txt
```

说明：

- `A0(DynPillarVFE)` 需要 `torch-scatter`。
- 若只跑 `Radar7PillarVFE`（A1/A2/A3/A4），可不依赖 `torch-scatter`。

## 3. 数据准备

```powershell
# 检查目录与 split
.\scripts\prepare_vod_radar_data.ps1

# 生成 infos
.\scripts\create_infos.ps1
```

默认读取：`E:\毕设\code\vod-min`

## 4. 训练与验证

### 4.1 主配置训练

```powershell
.\scripts\train_centerpoint_vod_radar.ps1 -Config configs/model_configs/vod_centerpoint_radar_v1_5.yaml -ExtraTag v1_5
```

### 4.2 主配置验证

```powershell
.\scripts\eval_centerpoint_vod_radar.ps1 -Config configs/model_configs/vod_centerpoint_radar_v1_5.yaml -Ckpt <your_ckpt_path>
```

## 5. Decode/NMS 调参

```powershell
$env:PYTHONPATH = "$PWD;$PWD\external\OpenPCDet"
python -m pcdet_ext.utils.decode_tuning \
  --cfg configs/model_configs/ablation_a2_radar7pillar_vel.yaml \
  --ckpt <your_ckpt_path> \
  --output-json outputs/ablations/decode_tuning.json \
  --output-csv outputs/ablations/decode_tuning.csv
```

支持：

- `class_agnostic_nms`
- `class_specific_nms`
- `circle_nms`（当前 OpenPCDet 版本未完整开放 decode 端，脚本会标记 unavailable）

## 6. 消融实验

```powershell
# 仅打印计划，不执行
.\scripts\run_ablation_suite.ps1 -DryRun

# 执行 A0~A4 + feature/velocity 子实验
.\scripts\run_ablation_suite.ps1 -RunTrain
```

结果输出：

- `outputs/ablations/*.json`
- `outputs/ablations/ablation_summary.csv`
- `outputs/ablations/ablation_summary.md`

## 7. 官方评估适配

先确保已有 OpenPCDet 验证输出的 `result.pkl`，再导出：

```powershell
.\scripts\export_official_eval.ps1 -ResultPkl <path_to_result.pkl>
```

导出目录：`outputs/official_eval/predictions/*.txt`

可选：通过 `-DevkitCmd` 传入官方 devkit 命令模板（使用 `{pred_dir}` 占位符）。

## 8. 关键实现点

### 8.1 VodRadarDataset

- 读取 VoD 的 `radar_5frames`/`label_2`/`calib`/`ImageSets`
- 生成 `vod_radar_infos_train/val/trainval/test.pkl`
- 支持 KITTI-style 评估兼容字段

### 8.2 Radar7PillarVFE

- 输入点特征：`[x, y, z, rcs, v_r, v_r_comp, time]`
- 几何/反射分支：`xyz, rcs, cluster offset, pillar center offset, distance`
- 运动分支：`v_r, v_r_comp, time, range`
- 双流聚合（max+mean）后融合输出

### 8.3 速度监督

- 主方法：`u_i^T v ≈ v_r_comp,i` 的鲁棒 2D IRLS 求解
- 可靠性判定：最少点数、条件数、残差、速度幅值
- 回退：heading 投影弱监督
- 输出：`vx, vy, vel_weight`（弱监督权重更低）

### 8.4 CenterHead 增强

- 支持 `USE_LOCAL_MAX` heatmap 峰值抑制
- 支持 `class_agnostic_nms` 别名
- 支持 `circle_nms` 后处理接口
- 回归损失支持按目标加权（用于 velocity weak/strong）

## 9. 已知限制

- 当前 OpenPCDet 上游 `decode_bbox_from_heatmap` 的 `circle_nms` 路径仍标注未完全验证，因此本工程默认不在 decode 阶段启用该路径。
- `vel_fit_residual_mean` 统计来自数据侧样本聚合，不是 dense head 内部重算值。
- 未重写 OpenPCDet 主训练框架，EMA 未作为默认训练组件接入（保留 scheduler + 可复现配置）。

## 10. 建议下一步

1. 先跑 `A1 -> A2`，确认 `loss_vel` 和 `num_valid_vel_boxes` 曲线。  
2. 再跑 `decode_tuning` 固化最优阈值，更新到 `A3/A4`。  
3. 最后批量跑 `run_ablation_suite.ps1` 生成论文表格。  
