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

## Checkpoint EEG tự huấn luyện

`eeg_vigilance_v2/model.pt` là mô hình fatigue được chọn bằng validation trên dữ liệu
Driver Fatigue EEG (Min et al., 2017; CC BY 4.0). Có trọng số, biến đổi công suất tương đối,
chuẩn hóa train-only, ngưỡng, split theo người, fingerprint dữ liệu và thông số trích EEG.
Các checkpoint Ridge, Logistic, SVM và TCN đối chứng cũng nằm trong thư mục này.

`metrics.json`, `predictions_test.csv`, `selection.json`, `evaluation.png` lưu kết quả cùng
phiên train. `eeg_vigilance_v1/` giữ baseline phát triển ban đầu. Toàn bộ artifact được
giữ cục bộ và ignored, không đưa lên Git. Xem `docs/05_model_usage.md` để tái lập và
`docs/06_model_results.md` để biết kết quả, đặc biệt lỗi ngưỡng khi đổi người dùng.
