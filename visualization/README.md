# Visualization Site

该目录独立于 `baseline/` 与 `improve-v1/`，用于网页可视化展示实验结果。

## 页面结构

- 主界面：`/`
  - 只做导航与概览
  - 提供 Baseline、V1、50/50 播放器入口
- Baseline 结果页：`/baseline-results`
  - baseline 训练曲线与最新指标
  - 数据统计
  - train/test 标签播放
- V1 结果页：`/v1-results`
  - centerpoint_v1 与消融实验切换
  - `mean_f1` 曲线、`train/val loss` 曲线
  - 最新指标表与相对 baseline 对比
- 50/50 播放器：`/split-player`
  - 上半 train、下半 test
  - 每个播放器左图像+标签，右 BEV

## 启动

```powershell
cd E:\毕设\code\4d-radar
python .\visualization\server.py --host 127.0.0.1 --port 8090
```

访问：

- `http://127.0.0.1:8090/`
- `http://127.0.0.1:8090/baseline-results`
- `http://127.0.0.1:8090/v1-results`
- `http://127.0.0.1:8090/split-player`

## 数据来源

- Baseline history: `baseline/outputs/vod_baseline/history.json`
- Improve-v1 history:
  - `improve-v1/outputs/vod_centerpoint_radar_v1/history.json`
  - `improve-v1/outputs/ablation_baseline_like/history.json`
  - `improve-v1/outputs/ablation_baseline_post/history.json`
  - `improve-v1/outputs/ablation_feature_enhanced/history.json`
- 图片：`..\vod-min\lidar\training\image_2`
- 标签：`..\vod-min\lidar\training\label_2`
- 雷达点：`..\vod-min\radar_5frames\training\velodyne`
- split：`..\vod-min\lidar\ImageSets\{train,val,test,train_val}.txt`

## API

- `GET /api/health`
- `GET /api/summary`
- `GET /api/history?exp=baseline|centerpoint_v1|ablation_baseline_like|ablation_baseline_post|ablation_feature_enhanced`
- `GET /api/experiments`
- `GET /api/samples?split=train|test|val|train_val&limit=300&offset=0`
- `GET /api/labels/{sample_id}`
- `GET /api/radar/{sample_id}?max_points=8000&x_min=0&x_max=60&y_min=-30&y_max=30&color_by=velocity|label`
- `GET /data/image/{sample_id}.jpg`

