# 4D Radar 统一实验看板

`visualization/` 已从“baseline/v1 写死页面”升级为**实验注册表驱动**平台，统一接入：

- `baseline/`
- `improve-v1/`
- `improve-v1.5/`

支持：

- 首页总览（家族视图）
- 实验总览筛选
- 实验详情（quick + official 双轨）
- 多实验对比（指标 + 配置差异 + 曲线）
- 样本级查看器（图像 + BEV，多模型 overlay/grid）
- improve-v1.5 专题（A0~A4、特征/速度消融、decode tuning）

## 目录结构

```text
visualization/
├─ server.py
├─ config/
│  ├─ app_config.yaml
│  └─ experiments_registry.yaml
├─ data_adapters/
│  ├─ baseline_adapter.py
│  ├─ improve_v1_adapter.py
│  ├─ improve_v1_5_adapter.py
│  ├─ sample_predictions.py
│  └─ common_metrics.py
├─ services/
│  ├─ registry_service.py
│  ├─ experiment_service.py
│  ├─ compare_service.py
│  └─ sample_viewer_service.py
├─ api/
│  ├─ experiments.py
│  ├─ compare.py
│  ├─ samples.py
│  └─ metrics.py
└─ web/
   ├─ index.html
   ├─ experiments.html
   ├─ experiment_detail.html
   ├─ compare.html
   ├─ sample_viewer.html
   ├─ v1_5_results.html
   ├─ official_eval.html
   ├─ ablations.html
   └─ *.js / styles.css
```

## 页面说明

- `/`：实验总览首页（按 family 聚合）
- `/experiments`：实验列表页（支持筛选/排序）
- `/experiments/<id>`：实验详情页
- `/compare`：多实验对比页
- `/sample-viewer`：样本级对比查看器
- `/v1_5-results`：improve-v1.5 专题页
- `/official-eval`：official 指标查看页
- `/ablations`：ablation 汇总页

兼容旧路由：

- `/baseline-results` -> `baseline_main` 详情
- `/v1-results` -> `improve_v1_main` 详情
- `/split-player` -> 新 `sample-viewer`

## API

- `GET /api/health`
- `GET /api/app-info`
- `GET /api/family-overview`
- `GET /api/experiments`
- `GET /api/experiments/<id>`
- `GET /api/experiments/<id>/metrics`
- `GET /api/compare?ids=baseline_main,improve_v1_main,...`
- `GET /api/samples?split=val&limit=300&offset=0`
- `GET /api/samples/<sample_id>`
- `GET /api/samples/<sample_id>/predictions?exp=baseline_main,improve_v1_5_main`
- `GET /api/ablations?family=improve-v1.5`
- `GET /api/decode-tuning?exp=improve_v1_5_main`
- `GET /data/image/<sample_id>.jpg`

## 实验注册表机制

核心文件：`config/experiments_registry.yaml`。

每个实验条目至少建议包含：

- `id`
- `display_name`
- `family` (`baseline` / `improve-v1` / `improve-v1.5`)
- `status` (`active` / `draft` / `archived`)
- `description`
- `history_path`
- `latest_metrics_path`
- `config_snapshot_path`
- `official_eval_path`
- `ablation_summary_path`
- `decode_tuning_path`
- `prediction_dir`
- `quick_metrics_type`
- `official_metrics_type`
- `tags`
- `color`
- `sort_order`

> 所有页面与 API 都基于 registry 动态发现实验，不再写死实验名。

## 如何新增一个实验

1. 在 `config/experiments_registry.yaml` 增加新条目。
2. 路径尽量写相对 `project_root` 的相对路径（在 `app_config.yaml` 中配置）。
3. 保证至少有一个 quick 指标来源：
   - `history.json`
   - OpenPCDet `train_*.log`
   - ablation 汇总 CSV
4. 如果有 official 结果，填 `official_eval_path`。
5. 如果有样本预测 JSON，填 `prediction_dir`。
6. 刷新页面或重启 server 后自动可见。

## 样本预测接入

统一入口：`data_adapters/sample_predictions.py`。

当前支持两类来源：

1. `prediction_dir/<sample_id>.json`
2. OpenPCDet `result.pkl`（自动解析并缓存为标准化 JSON）

自动缓存目录：

```text
visualization/artifacts/viz_samples/<exp_id>/<sample_id>.json
```

标准化样本结构（前端消费）：

- `sample_id`
- `gt_boxes`
- `pred_boxes`
- `label`
- `score`
- `box_lidar`
- `image_url`
- `radar_points`

## 本地启动

```powershell
cd E:\毕设\code\4d-radar
python .\visualization\server.py --host 127.0.0.1 --port 8090
```

打开：

- `http://127.0.0.1:8090/`

## 已接入结果说明

- `baseline_main`：`history.json + latest_metrics`
- `improve_v1_main`：`history.json`
- `improve_v1.5_main`：OpenPCDet `train log + official rows + ablation summary + decode tuning`
- `improve-v1.5` A0~A4、feature、velocity 通过 `ablation_summary.csv` 聚合展示

## 降级策略

以下情况不会导致页面崩溃：

- 仅有 quick，无 official
- 无 sample prediction 文件
- 无 history 曲线
- registry 注册了但产物路径不存在
- ablation 尚未跑完

页面会显示“暂无数据”或“官方评估暂缺”。

## 已知限制

- 样本图像目前仅读取 `.jpg`。
- 2D 图像框当前显示 GT 结构化信息，未做多模型 2D 投影框叠加。
- `result.pkl` 解析依赖 `frame_id` 与数据集 sample id 一致。
- decode tuning 若无 `decode_tuning.json/.csv`，只显示空态。
