# Visualization Site

该目录独立于 `baseline/`，用于展示训练结果与数据可视化。

## 功能

- 主页面 Dashboard：`/`
  - 训练历史曲线与指标
  - 数据集统计
  - train/test 两个分区播放器
- 子页面 50/50 播放器：`/split-player`
  - 上半：`train` 播放器
  - 下半：`test` 播放器
  - 每个播放器都为左半屏 `image + label boxes`，右半屏 `radar BEV`
  - 两个播放器独立控制（FPS / step / slider）

## 启动

```powershell
cd E:\毕设\code\4d-radar
python .\visualization\server.py --host 127.0.0.1 --port 8090
```

打开：

- Dashboard: `http://127.0.0.1:8090/`
- 50/50 子页面: `http://127.0.0.1:8090/split-player`

## 数据来源

- Baseline 历史：`baseline/outputs/vod_baseline/history.json`
- 图片：`..\vod-min\lidar\training\image_2`
- 标签：`..\vod-min\lidar\training\label_2`
- 雷达点：`..\vod-min\radar_5frames\training\velodyne`
- split：`..\vod-min\lidar\ImageSets\{train,val,test,train_val}.txt`

## API

- `GET /api/health`
- `GET /api/summary`
- `GET /api/history`
- `GET /api/samples?split=train|test|val|train_val&limit=300&offset=0`
- `GET /api/labels/{sample_id}`
- `GET /api/radar/{sample_id}?max_points=8000&x_min=0&x_max=60&y_min=-30&y_max=30&color_by=velocity|label`
- `GET /data/image/{sample_id}.jpg`
