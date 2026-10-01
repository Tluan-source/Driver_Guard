# A1 audit và CAM-01: kết quả development

Cập nhật 01/10/2026. Phiên này đã hoàn thành audit kỹ thuật A1, chạy đối chứng
CAM-01 và chuẩn bị gói gán nhãn A2. **Chưa có cải thiện chất lượng được chứng
minh; giữ camera v1 làm checkpoint chính.** Mốc A chưa đạt 85% và chưa hoàn tất.

## 1. Audit dữ liệu

Báo cáo máy đọc được: [audit.json](results/camera_development_a1/audit.json).

| Kiểm tra | Kết quả |
|---|---|
| Hash và thông số media video | 354/354 khớp manifest |
| Hash feature NPZ | 588/588 khớp manifest |
| Người giữa train/validation/test | Không giao nhau |
| ID recording giữa các split | Không giao nhau; ID suy ra từ đường dẫn nguồn |
| Exposure augmentation | Chỉ có trong train |
| Video trùng hash | Không phát hiện |
| Gần trùng train/validation | Không có ứng viên dHash ≤2 trên frame giữa |
| Frame hợp lệ train, gồm augmentation | 97,85% |
| Coverage cửa sổ train sau warmup | 97,79%; từ chối 62/2.808 cửa sổ |
| Coverage cửa sổ validation sau warmup | 100%; 360/360 cửa sổ |

Không có blocker kỹ thuật trong các kiểm tra này. Hash và heuristic một frame
không chứng minh toàn bộ corpus không trùng nội dung hoặc nhãn đúng. Không đọc
mảng NPZ hoặc tính thêm predictions test để phát triển model.

Đã xem bốn contact sheet đầu/giữa/cuối của 40 clip development. Các quan sát
là ưu tiên kiểm tra, **chưa phải nhãn được người xác nhận**:

- Nhiều clip nhãn buồn ngủ vẫn có mắt mở ở ba frame mẫu. Đây có thể là trạng
  thái buồn ngủ chưa có biểu hiện rõ; không được đổi nhãn chỉ vì mắt mở.
- Kính, góc đầu, kích thước mắt và biểu cảm thay đổi theo người; cần xem video
  và lỗi theo người trước khi quy lỗi cho model hoặc nhãn.
- `S48_10_0004_04` có ánh sáng yếu/tư thế ngửa đầu; raw-valid fraction chỉ 32%.
  Không hạ quality gate để ép model cho score.
- Độ sáng nguồn trung bình lớp tỉnh/buồn ngủ khác nhau: train 101,60/93,91;
  validation 92,95/75,86. Đây là yếu tố nhiễu cần theo dõi; CAM-01 không đưa
  brightness trực tiếp vào input, nhưng geometry vẫn chịu ảnh hưởng ánh sáng.

Nhãn hiện tại vẫn theo recording, clip chỉ 10 giây/10 FPS, crop mặt 224×224.
Chưa có ground truth onset, video đêm thật, EEG đồng bộ hay cohort camera mới.

## 2. CAM-01 so với v1

CAM-01 dùng 79 thống kê temporal có mask của mắt/miệng/đầu và logistic regression.
Scaler chỉ fit train; thử C = 0,01/0,1/1/10 bằng 5 fold tách người trong train.
Chọn C=0,01 theo weighted OOF log loss; ngưỡng 0,45 chọn trên validation.
Context 50 mẫu ở 10 Hz, stride đánh giá 10 mẫu, quality gate giữ nguyên v1.

So sánh dưới đây đều là **validation window**, cùng 10 người và 360 cửa sổ:

| Metric | Camera v1 GRU | CAM-01 temporal logistic |
|---|---:|---:|
| Accuracy | 61,94% | 61,11% |
| Macro-F1 pooled | 61,88% | 60,72% |
| Macro-F1 trung bình theo người | 56,42% | 55,91% |
| Recall buồn ngủ | 66,11% | 71,11% |
| Specificity | 57,78% | 51,11% |
| AUROC | 0,6528 | 0,6073 |
| Coverage sau warmup | 100% | 100% |

CAM-01 tăng recall nhưng giảm specificity và các chỉ số chính. Không chọn
CAM-01 thay v1. Kết quả không chứng minh geometry không đủ thông tin; mới thử
logistic với một bộ summary trên corpus yếu nhãn.

Bootstrap 300 lần theo người, CI 95% macro-F1 CAM-01: 50,17–70,29%. Đây là CI
validation có điều kiện theo cấu hình/ngưỡng đã chọn, không phải CI final test.
Toàn run mất 19,96 giây trên CPU. RTX 3050 hiện có nhưng Torch đang là bản CPU.
Không chạy thêm test v1. Chi tiết:
[camera_cam01_development.json](results/camera_cam01_development.json).

## 3. Checkpoint dùng được để đối chứng

Artifact cục bộ: `models/camera_cam01_temporal_logistic/` gồm `model.pt`,
`metrics.json`, `predictions.csv`, `clip_predictions.csv`, `configurations.csv`.
Checkpoint SHA-256:
`3d09bafffba7a96b3384282d354c154fab2d9a65ace3d97a1a99c488d29cb069`.

Score export khớp sklearn với sai lệch tối đa 3,33e-16. Video development
`01_0_0_00.mp4` cho 51 score từ 4,9 giây, khớp NPZ stride=1 với sai lệch 0.
P50/P95 perception + model + luật: 37,72/44,30 ms, 100 frame 224×224.
Đây chưa phải capture-to-result webcam, benchmark ≥15 FPS hoặc nghiệm thu đêm.

Chạy webcam v1 để test luồng nhận diện mặt, mắt, mũi, miệng và cảnh báo:

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.camera_cli run --checkpoint models/camera_drowsiness_v1/model.pt --source 0 --show --jsonl outputs/camera_live.jsonl
```

Đổi checkpoint sang CAM-01 để so hành vi hai model, không coi score là xác suất
đã calibration. Cả hai chưa được EEG huấn luyện để cảnh báo sớm.

Tái tạo CAM-01 trong thư mục mới:

```powershell
.\.venv\Scripts\python.exe -m driverguard.learning.camera_baseline --manifest data/features/camera_uta_pilot/manifest.json --out models/camera_cam01_reproduce --seed 42 --stride 10
```

## 4. Task đang mở

1. Rà đầy đủ 40 video và xác nhận interval nhãn A2 theo
   [protocol](16_camera_annotation_protocol.md). Tách yếu tố nhìn thấy được
   khỏi trạng thái buồn ngủ sinh lý; không sửa nhãn dựa vào dự đoán model.
2. Tạo cohort camera độc lập người/nguồn; chốt quyền sử dụng và split trước
   khi tải/thu lớn. UTA gốc có thể thêm context nhưng không tạo người mới nếu
   trùng với 59 người đã dùng.
3. Sau khi xem kết quả A2, chạy CAM-02 GRU cải tiến và CAM-03 TCN trên cùng
   protocol; giới hạn cấu hình theo roadmap, ghi cả kết quả âm và kiểm tra seed.
4. Chỉ chuyển sang nghiệm thu ban đêm/EEG sau khi các điều kiện tương ứng đủ.

Code, protocol và báo cáo lưu Git. Trọng số, video, feature và biểu mẫu người
gán nhãn lưu cục bộ; chưa tải dataset mới trong phiên này.

Kiểm tra code: toàn suite hiện có 290 tests đã pass, thêm 8 tests annotation
đã pass; tổng 298 tests. Ruff trên các file thay đổi và `git diff --check`
đều pass. Các kiểm tra bao gồm loader không mở test NPZ, export/inference,
quality gate, provenance, video checksum và interval annotation không hợp lệ.
