# models/

Không commit file model vào git. Tải về bằng:

```bash
python scripts/download_models.py
```

| File | Nguồn | License | Dùng cho |
|---|---|---|---|
| `face_landmarker.task` | MediaPipe Face Landmarker (float16) | Apache-2.0 (MediaPipe) — kiểm tra model card | 478 landmark, 52 blendshape, ma trận tư thế |
| `efficientdet_lite0.tflite` | MediaPipe Object Detector (COCO) | Apache-2.0 — kiểm tra model card | Phát hiện `cell phone` |

Model YOLO (Ultralytics) **không** dùng mặc định vì giấy phép AGPL-3.0 — nếu muốn thử, ghi rõ trong báo cáo và không đưa vào bản thương mại.
