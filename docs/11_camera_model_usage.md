# Sử dụng mô hình camera giai đoạn 1

Đây là prototype camera nhận diện **trạng thái buồn ngủ hiện tại** từ chuỗi
đặc trưng khuôn mặt. MediaPipe nhận diện khuôn mặt và các điểm mắt, mũi, miệng;
GRU đã học đặc trưng mắt, miệng và tư thế đầu theo thời gian. Nhánh EEG chưa
được dùng để giám sát model này. Model cảnh báo sớm theo onset EEG cần bộ
camera–EEG đồng bộ cùng người, cùng phiên và sẽ được huấn luyện sau.

Chạy các lệnh PowerShell từ `D:\code\DriverGuard`, dùng `.venv` của dự án.
Checkpoint, video và NPZ được lưu cục bộ, không được đưa vào Git.

## Dữ liệu và hợp đồng model

- Nguồn: [UTA-RLDD face-cropped trên Kaggle](https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-videos-cropped-by-faces).
  Corpus chọn 59 người, 354 clip RGB 10 giây, ảnh crop 224×224 ở 10 FPS;
  mỗi người có ba clip nhãn tỉnh và ba clip nhãn buồn ngủ. Subject 42 bị loại
  vì thiếu lớp buồn ngủ trong bản derivative. Nhãn nguồn 0 → tỉnh, 10 → buồn ngủ;
  nhãn trung gian không được dùng.
- Chia cố định theo người: train 39, validation 10, test 10. Cùng người và mọi
  biến thể ánh sáng của người đó ở cùng một tập. Train có thêm exposure 0,6;
  validation và test giữ mức sáng nguồn. Ảnh tối nhân tạo không thay thế video đêm thật.
- Đầu vào GRU: 50 mẫu quá khứ, tương đương khoảng 5 giây ở 10 Hz; 19 đặc trưng
  gồm chín giá trị hình học/blendshape, chín mask hợp lệ và một mask mặt hợp lệ.
  GRU có hidden size 32. Thứ tự đặc trưng, scale, chất lượng ảnh, tiền xử lý tối
  và SHA-256 Face Landmarker được giữ trong checkpoint.
- Nhãn được gán theo cả video, không xác nhận từng khung hình hoặc onset.
  Ngưỡng quyết định và epoch được chọn bằng validation; test không chọn model/ngưỡng.
  Score trong [0,1] là đầu ra model **chưa calibration**, không diễn giải thành
  xác suất chính xác tài xế buồn ngủ.

Các artifact chính trong `models/camera_drowsiness_v1/`:

| File | Vai trò |
|---|---|
| `model.pt` | Checkpoint camera-only và hợp đồng đầu vào |
| `metrics.json` | Protocol, ngưỡng, split và metrics train/validation/test |
| `predictions.csv` | Score từng cửa sổ, giữ cả mẫu bị từ chối |
| `clip_predictions.csv` | Trung bình score của các cửa sổ hợp lệ theo clip |

## Chạy camera hoặc video

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.camera_cli run --checkpoint models/camera_drowsiness_v1/model.pt --source 0 --show --jsonl outputs/camera_live.jsonl
```

Hoặc dùng video cục bộ để kiểm tra luồng:

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.camera_cli run --checkpoint models/camera_drowsiness_v1/model.pt --source "D:\videos\driver_test.mp4" --show --realtime --jsonl outputs/camera_video.jsonl
```

Đầu ra bao gồm `camera_drowsiness_score`, `decision_threshold`, `model_status`,
`assessment_status`, `window_valid_fraction`, chất lượng/độ sáng mặt gốc,
`risk_level`, `reason_codes` và `alert`. `score_timestamp_ms` giúp kiểm tra tuổi
của score khi luồng camera chạy nhanh hơn tần số model. Preview thể hiện các
điểm mắt, mũi, miệng. JSONL chỉ chứa metadata, không ghi video.

Nhánh model cần khoảng 5 giây lịch sử trước khi có score. Sau đó chỉ đánh giá
khi ít nhất 60% mẫu trong cửa sổ hợp lệ, khung hiện tại hợp lệ và timestamp
đạt hợp đồng; mất dữ liệu quá 300 ms phải xây lại lịch sử. Khi không đủ điều
kiện, score là `null` và `assessment_status="cannot_assess"`; không coi đây là
dự đoán tỉnh táo. Nhánh luật nhanh vẫn theo dõi các dấu hiệu rõ như nhắm mắt kéo dài.

Tiền xử lý thiếu sáng dùng gamma/CLAHE có giới hạn, chỉ áp dụng khi ảnh gốc
còn đủ ánh sáng và tương phản. Chất lượng được đo trên **ảnh gốc**, nên tăng
sáng không biến dữ liệu không tin cậy thành bằng chứng hợp lệ. Ảnh gần như đen,
phẳng hoặc mặt không nhìn thấy sẽ dẫn tới không đánh giá được và
`sensor_degraded` sau thời gian cấu hình. Cổng sáng toàn ảnh có thể bỏ sót mặt
tối cạnh dashboard/cửa sổ sáng. Trong bóng tối hoàn toàn cần camera/chiếu sáng
NIR tương thích cùng bộ dữ liệu và kiểm chứng riêng; model RGB này chưa đáp ứng điều đó.

Runtime tách nhánh luật nhanh ở tốc độ nguồn tối đa 30 Hz và nhánh Face
Landmarker/GRU tại 10 Hz theo hợp đồng train. **Mục tiêu xử lý ≥15 FPS khác với
tần số score 10 Hz**. Kiểm tra headless trên clip 224×224 ở 10 FPS, 100 frame,
cho P50 37,17 ms và P95 47,33 ms; 51 score từ mốc 4,9 giây khớp hoàn toàn
với suy luận NPZ offline. Chưa kết luận đạt 15 FPS hoặc P95 ≤100 ms của toàn
hệ thống trên webcam/độ phân giải mục tiêu. CLI in
P50/P95 thời gian perception + model + luật; phép đo này chưa bao gồm capture,
queue, âm thanh cảnh báo hoặc preview. Tốc độ xử lý và số giây cảnh báo sớm
theo onset EEG là hai phép đo khác nhau.

## Dự đoán từ NPZ và tính metrics

Chọn một NPZ exposure nguồn đã được trích theo hợp đồng checkpoint:

```powershell
$cameraFeature = (Get-ChildItem data/features/camera_uta_pilot -Filter '*exp1.npz' | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe -m driverguard.learning.camera_cli predict --checkpoint models/camera_drowsiness_v1/model.pt --features $cameraFeature --out outputs/camera_scores.csv --stride 10
```

CSV có `subject`, `clip_id`, `end_ms`, `score`, `valid_fraction`, `reason`.
Score trống thể hiện từ chối đánh giá. Dùng `predictions.csv` hoặc
`clip_predictions.csv` của tập test để viết metrics; không trộn lại train/test,
không bỏ mẫu bị từ chối khỏi mẫu số coverage và không chọn ngưỡng trên test.
Score clip dùng cả clip để tổng hợp offline, không phải cảnh báo online tại một thời điểm.

```powershell
.\.venv\Scripts\python.exe scripts/report_camera_model.py --model-dir models/camera_drowsiness_v1 --out docs/results/camera_v1
```

Lệnh tạo `summary.json` và `heldout_metrics.png`: AUROC/AUPRC, confusion matrix,
balanced accuracy, macro F1 trên mẫu được chấp nhận, trung bình theo người,
coverage toàn bộ/theo lớp/theo người và khoảng tin cậy 95%. Bootstrap dùng
300 lần lấy mẫu cả người test, seed 42, ngưỡng quyết định cố định và ngưỡng
nhãn 0,5. Nếu một người không có mẫu hợp lệ, người đó vẫn nằm trong coverage;
CI pooled trên mẫu hợp lệ chỉ đại diện những người có score. Báo cáo giữ
SHA-256 checkpoint, CSV, source manifest và feature manifest.

Chưa có ground truth onset EEG nên chưa thể tính lead time. Clip ngắn cân
bằng lớp cũng chưa đủ để kết luận cảnh báo sai mỗi giờ khi lái xe thật, độ
chính xác ban đêm hoặc khả năng tổng quát sang camera/người/xe độc lập.
Kết quả hiện tại: test clip AUROC 0,6444, balanced accuracy 0,55,
recall 0,70 và specificity 0,40. Xem [báo cáo đầy đủ](12_camera_model_results.md)
trước khi diễn giải score hoặc cảnh báo.

## Tái tạo dữ liệu và train

Giữ các artifact v1 để đối chiếu. Lệnh train dưới đây dùng thư mục mới, không
ghi đè checkpoint hiện tại; đây là thao tác tái tạo nghiên cứu, không cần chạy
trước khi test model v1.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_camera_uta.py --download --out data/raw/camera_uta_pilot --clip-length 10 --clips-per-class 3 --seed 42 --workers 4
.\.venv\Scripts\python.exe -m driverguard.learning.camera_extract --manifest data/raw/camera_uta_pilot/manifest.json --out data/features/camera_uta_pilot --workers 2 --train-exposures 1.0 0.6
.\.venv\Scripts\python.exe -m driverguard.learning.camera_train --manifest data/features/camera_uta_pilot/manifest.json --out models/camera_drowsiness_v1_reproduce --epochs 30 --patience 7 --seed 42 --batch-size 64 --stride 10
```

Feature cache được kiểm tra checksum và extraction signature. Thay cấu hình
chất lượng, tiền xử lý hoặc Face Landmarker cần trích lại dữ liệu vào thư mục
mới và train model có hợp đồng tương ứng. File lớn được giữ cục bộ; Git lưu
code, protocol và báo cáo để tái tạo.
