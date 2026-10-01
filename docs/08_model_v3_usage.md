# Sử dụng mô hình EEG v3

Phiên này tập trung vào mô hình EEG. Agents hỗ trợ và IoT sẽ được triển khai sau.

## Đầu vào và đầu ra

Checkpoint robust: `models/eeg_vigilance_v3_1/model.pt`. Checkpoint tham chiếu
dùng tiền xử lý cũ và calibration mới: `models/eeg_vigilance_v3_1/legacy_model.pt`.
Mỗi cửa sổ dùng 4 giây EEG,
30 kênh theo đúng thứ tự trong metadata và 5 dải tần. Điểm càng cao thể hiện
model càng nghiêng về trạng thái fatigue trong thí nghiệm; đây không phải
xác suất chẩn đoán lâm sàng. Ngưỡng quyết định cố định là 0,5.

Tiền xử lý `robust_v1` kiểm tra chất lượng điện cực, dùng median reference
và nội suy điện cực lỗi trong giới hạn cho phép. Cửa sổ không đạt chất lượng
được giữ đúng timestamp nhưng không nhận score hoặc quyết định. Dữ liệu
`fatigue_eeg` cũ không tương thích với checkpoint này.

## Chạy thử

Chạy tại thư mục gốc dự án trong PowerShell:

```powershell
.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v3_1/model.pt --features data/features/fatigue_eeg_robust_v1/subject_03_normal.npz --out outputs/eeg_v3_predictions.csv
```

Hoặc dùng bản ghi CNT gốc, không cần nhãn:

```powershell
.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v3_1/model.pt --cnt "data/raw/driver_fatigue_eeg/3/3/Normal state.cnt" --out outputs/eeg_v3_cnt_predictions.csv
```

Chạy checkpoint tham chiếu với feature cũ để so sánh:

```powershell
.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v3_1/legacy_model.pt --features data/features/fatigue_eeg/subject_03_normal.npz --out outputs/eeg_v3_reference_predictions.csv
```

CSV chứa `window_index`, `window_end_seconds`, `prediction`,
`reduced_vigilance_pred`, `quality_valid`, `sensor_status`, `abstention_reason`.
Hai cột dự đoán để trống khi `quality_valid=0`; trạng thái là `SENSOR_DEGRADED`.
Không thay ô trống bằng 0 khi tính metrics hoặc nối với hệ thống cảnh báo.

## Viết Metrics

Dùng `models/eeg_vigilance_v3_1/robust_nested_loso_predictions.csv` để tính metrics
phát triển. Mỗi dòng có `target`, người tham gia, bản ghi, chỉ số cửa sổ,
score và quyết định của lượt kiểm tra giữ người đó ngoài tập huấn luyện.

```python
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

df = pd.read_csv("models/eeg_vigilance_v3_1/robust_nested_loso_predictions.csv")
accepted = df[df["accepted"].eq(True)].copy()
print("Coverage:", len(accepted) / len(df))
print("AUROC:", roc_auc_score(accepted["target"], accepted["prediction"]))
print("Balanced accuracy:", balanced_accuracy_score(
    accepted["target"], accepted["reduced_vigilance_pred"]))
```

Tính thêm metrics riêng từng `subject`, rồi lấy trung bình giữa các người.
JSON báo cáo có AUROC, AUPRC, balanced accuracy, macro F1, sensitivity,
specificity, log loss, Brier score, confusion matrix và khoảng tin cậy bootstrap
theo người. `coverage_adjusted` coi mỗi abstention là sai so với lớp thật,
giúp đánh giá đồng thời độ chính xác và khả năng trả lời của model.

Checkpoint triển khai được fit trên toàn bộ cohort phát triển sau khi chọn cấu
hình bằng LOSO nội bộ. Dự đoán của checkpoint trên dữ liệu đã train chỉ dùng
để thử luồng suy luận; không dùng làm kết quả test. Lệnh `evaluate` dành cho
split test đóng băng của v2 và sẽ từ chối checkpoint v3 không có split đó.

## Tái Lập

Giữ bộ feature và model cũ, tạo một thư mục đầu ra mới:

```powershell
.venv\Scripts\python.exe scripts/prepare_fatigue_eeg.py --raw data/raw/driver_fatigue_eeg --out data/features/fatigue_eeg_robust_reproduce --preprocessing robust_v1
.venv\Scripts\python.exe -m driverguard.learning.cli improve --baseline-data data/features/fatigue_eeg --data data/features/fatigue_eeg_robust_reproduce --out models/eeg_vigilance_v3_1_reproduce --bootstrap 1000
```

`protocol.json` ghi grid cấu hình, seed, phiên bản thư viện và hash source trước
khi đánh giá. `deployment_selection.json` ghi lựa chọn từ các lượt nội bộ.
Hai báo cáo `legacy_nested_loso.json` và `robust_nested_loso.json` dùng cùng
quy tắc đánh giá; `summary.json` chứa so sánh ghép cặp theo người.

Bản v3 đầu tiên trong `models/eeg_vigilance_v3` được giữ làm lịch sử thử nghiệm.
Bản v3.1 sửa lỗi cắt điểm LDA trước calibration, dùng trực tiếp raw decision
margin. Việc sửa lỗi không đổi ba cấu hình ứng viên hoặc ngưỡng 0,5. Kết quả
không đảm bảo tăng cho mọi người; xem báo cáo thực nghiệm trước khi chọn model.

Dữ liệu gốc, features và checkpoint được Git ignore. Code, kiểm thử và tài liệu
được lưu trên Git; muốn tái lập trên máy khác cần tải dữ liệu và chạy pipeline.

## Giới Hạn

Cohort chỉ gồm 12 nam trẻ trong simulator và nhãn normal/fatigue ở mức bản ghi.
Một phần cohort đã được dùng để nghiên cứu baseline trước đó. Nested LOSO giữ
người kiểm tra ngoài quá trình fit từng lượt, nhưng vẫn là đánh giá phát triển
trên cùng nguồn dữ liệu. Cần cohort độc lập để chứng minh khả năng tổng quát.
Hiện chưa thể đo onset microsleep, độ trễ phát hiện hoặc cảnh báo sai mỗi giờ.
EEG từ thiết bị ít kênh hoặc montage khác cần hợp đồng đầu vào và huấn luyện riêng.
