# Sử dụng mô hình EEG mệt mỏi

Chạy lệnh PowerShell từ thư mục gốc `D:\code\DriverGuard`. Môi trường `.venv` hiện tại
dùng Python 3.10; dự án hỗ trợ Python 3.10-3.12. Khi tạo môi trường mới, chạy
`py -3.10 -m venv .venv`, rồi cài đặt:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

## Dữ liệu và checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/download_driver_fatigue_eeg.py --out data/raw/driver_fatigue_eeg
.\.venv\Scripts\python.exe scripts/prepare_fatigue_eeg.py --raw data/raw/driver_fatigue_eeg --out data/features/fatigue_eeg --window-seconds 4
.\.venv\Scripts\python.exe -m driverguard.learning.cli train --data data/features/fatigue_eeg --out models/eeg_vigilance_v2 --epochs 60 --context 8 --seed 42 --threads 4 --bootstrap 300 --feature-transform relative_log_power
```

Script tải bản gốc Figshare CC BY 4.0 của bộ Driver Fatigue EEG có mirror trên Kaggle;
không cần Kaggle token. Xem nguồn và checksum trong `data/raw/driver_fatigue_eeg/provenance.json`.
`prepare` ghi một file NPZ cho mỗi bản ghi, `sessions.json` và `dataset_metadata.json`.
Nhãn bình thường = 0, mệt mỏi = 1, được gán theo cả bản ghi.

Checkpoint bàn giao cuối được chọn trên validation nằm tại **`models/eeg_vigilance_v2/model.pt`**.
Pipeline so sánh tám họ ứng viên: mean baseline, Ridge, Logistic tuyệt đối/tương đối,
SVM tuyệt đối/tương đối và TCN một cửa sổ/ngữ cảnh. Không chọn họ model theo test.
`v1` giữ kết quả công suất tuyệt đối chọn mô hình hằng số để đối chiếu nghiên cứu;
hãy dùng `v2` cho dự đoán. `train` từ chối ghi vào thư mục đã có `model.pt`.
Để train lại, dùng `--out models/eeg_vigilance_v3`; giữ báo cáo test cũ để so sánh minh bạch.

`--feature-transform` nhận `absolute` hoặc `relative_log_power`, điều khiển Ridge/TCN.
Nếu bỏ cờ, nhiệm vụ fatigue mặc định dùng `relative_log_power`: công suất từng dải chia
tổng năm dải của cùng kênh, lấy `log10`, rồi áp dụng scaler train. Logistic/SVM luôn so sánh
cả hai cách biểu diễn với scaler riêng. Biến đổi và scaler được lưu trong từng checkpoint.

## Dự đoán

Dùng đặc trưng đã chuẩn bị:

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v2/model.pt --features data/features/fatigue_eeg/subject_01_normal.npz --out outputs/eeg_normal.csv
```

Hoặc đọc trực tiếp CNT theo đúng bộ kênh và cách trích đặc trưng của checkpoint:

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v2/model.pt --cnt "data/raw/driver_fatigue_eeg/1/1/Normal state.cnt" --out outputs/eeg_cnt.csv
```

CSV có `window_index`, `window_end_seconds`, `prediction`, `reduced_vigilance_pred`.
`prediction` là score mệt mỏi trong [0, 1]; cột tên `reduced_vigilance_pred` là quyết định
0/1 tại ngưỡng của checkpoint, với **1 = trạng thái mệt mỏi** trong thí nghiệm này.
Đầu ra JSON trên terminal ghi target và ngưỡng sử dụng. Score chưa được hiệu chỉnh xác suất.

`--features` nhận NPZ với mảng `x` hoặc NPY `[time, 150]` hữu hạn, theo đúng thứ tự
30 kênh × 5 dải tần và công thức `log10` bandpower của `dataset_metadata.json`.
Mỗi dòng tương ứng bốn giây. Vẫn đưa vào **đặc trưng công suất tuyệt đối đã lấy log10**;
predictor tự biến đổi theo checkpoint rồi áp dụng mean/std train. Không tính tương đối
hoặc chuẩn hóa trước lần nữa.
Mỗi file là một bản ghi độc lập; không nối nhiều người/bản ghi trước khi gọi predict.
Đầu bản ghi được đệm bằng cửa sổ đầu cho ngữ cảnh quá khứ còn thiếu.

Có thể dùng từ Python với cùng hợp đồng đầu vào:

```python
from driverguard.learning.model import VigilancePredictor

predictor = VigilancePredictor("models/eeg_vigilance_v2/model.pt")
scores = predictor.predict_file("data/features/fatigue_eeg/subject_01_normal.npz")
decisions = predictor.decisions(scores)
threshold = predictor.meta["decision_threshold"]
# Với mảng đặc trưng x có shape [time, 150]: scores = predictor.predict(x)
```

## Evaluate và viết metrics

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.cli evaluate --checkpoint models/eeg_vigilance_v2/model.pt --data data/features/fatigue_eeg --out outputs/eeg_test_eval --bootstrap 300
```

Evaluate dùng các người test đã đóng băng trong checkpoint và kiểm tra hash/session để
tránh đánh giá nhầm dữ liệu. Xuất `metrics.json` và `predictions.csv`; CSV gồm `subject`,
`session_id`, `window_index`, `target`, `prediction`, `reduced_vigilance_pred`.
Chỉ đọc test để báo cáo kết quả; mọi lựa chọn model/ngưỡng phải dùng train/validation.

Các file trong thư mục model hỗ trợ phân tích:

| File | Nội dung |
| --- | --- |
| `selection.json` | Ứng viên thắng và metrics validation; AUROC trước, RMSE sau |
| `split.json` | 8 người train, 2 validation, 2 test; seed 42 |
| `dataset_manifest.json` | Session, số cửa sổ và hash dữ liệu |
| `metrics.json` | Metrics mọi ứng viên, theo người, CI của model được chọn |
| `predictions_test.csv` | Nhãn/score/quyết định của model được chọn trên test |
| `*_validation.csv`, `*_test.csv` | Dữ liệu dự đoán để tính metrics và vẽ biểu đồ |
| `*_history.json` | Loss train và RMSE validation theo epoch của TCN |

Confusion matrix có thứ tự `[[TN, FP], [FN, TP]]`. AUROC/AUPRC dùng score, các metrics
phân loại dùng ngưỡng đã lưu. RMSE/MAE đo score so với nhãn 0/1, không phải PERCLOS.
Kết quả bàn giao và diễn giải nằm tại [06_model_results.md](06_model_results.md).

Mô hình EEG chưa ghép với engine camera, Agents hoặc phần cứng IoT. Dữ liệu có 12 nam
giới trong mô phỏng; test chỉ có hai người và chưa có kiểm chứng trên đường thật.
Validation có hai người và đã dùng để phát triển biểu diễn/cấu hình; cần kiểm chứng
thêm bằng cross-validation và dữ liệu độc lập. Xem lịch sử phát triển trong kế hoạch nghiên cứu.
Git lưu mã/tài liệu; dữ liệu và checkpoint hiện bị `.gitignore` loại khỏi push thông thường.
