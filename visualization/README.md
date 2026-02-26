# Visualization Site

This folder is independent from `baseline/` and provides a web dashboard for:

- Baseline training result display (`history.json`)
- Dataset summary display
- Image + label sequence playback (video-like)

## Run

```powershell
cd E:\毕设\code\4d-radar
python .\visualization\server.py --host 127.0.0.1 --port 8090
```

Open:

```text
http://127.0.0.1:8090
```

## Data Sources

- Baseline history: `baseline/outputs/vod_baseline/history.json`
- Images: `..\vod-min\lidar\training\image_2`
- Labels: `..\vod-min\lidar\training\label_2`
- Splits: `..\vod-min\lidar\ImageSets\{train,val,train_val}.txt`

## APIs

- `GET /api/summary`
- `GET /api/history`
- `GET /api/samples?split=val&limit=300&offset=0`
- `GET /api/labels/{sample_id}`
- `GET /data/image/{sample_id}.jpg`

