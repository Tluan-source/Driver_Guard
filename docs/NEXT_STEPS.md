# Checklist công việc tiếp theo

Cập nhật 03/11/2026, dựa trên báo cáo khôi phục, protocol tuần 5 và docs dự án.
Tuần 6–7 bên dưới là kế hoạch đề xuất theo tài liệu, không phải kết quả đã hoàn thành.

## Mốc đã có

- [x] Khôi phục source, 30 checkpoint CV và 151 NPZ của 50 người development.
- [x] Kiểm hash/split, tái tính metrics và CI; replay CPU cả 30 checkpoint.
- [x] Kiểm runtime 120 giây bằng tín hiệu mô phỏng, mất mặt và reset.
- [x] Khôi phục dashboard/Model Lab; 307 test đạt, dashboard build thành công.

Accuracy development hiện tại: GRU 76,02%, DANN 75,42%; coverage 98,20%.
Mục tiêu chính theo protocol tuần 5 là **window accuracy >85%**, ngữ cảnh 120 giây;
macro-F1 và coverage báo riêng. Mục tiêu này chưa đạt.

## Ưu tiên 1 — có thể bắt đầu ngay

- [x] **Chốt bản phục hồi vào Git và sao lưu ngoài máy.** Rà diff, commit/push code,
  docs và test; sao lưu hai ZIP model/feature ở nơi lưu trữ riêng cùng checksum.
  Đầu ra: mốc Git có thể checkout và bản sao artifact có thể tải lại.
- [x] **Phân tích lỗi từ 30 run CV đã có.** Thống kê theo người, fold, seed, lớp,
  chất lượng và lý do từ chối; confusion matrix, recall/specificity, macro-F1 và
  coverage. Ghép đúng clip/timestamp khi so GRU với DANN. Đầu ra: CSV và báo cáo
  các nhóm lỗi cần ưu tiên; không dùng test gốc để lựa chọn cải tiến.
- [x] **Benchmark webcam thật trên máy đích.** Đo FPS, P50/P95 độ trễ, frame drops,
  warmup, mất mặt/camera, đổi model, reset và cảnh báo trên phiên liên tục. Đầu ra:
  log cấu hình thiết bị/camera và kết quả đo. Mục tiêu thiết kế trong docs là
  ≥15 FPS và P95 capture-to-result ≤100 ms; chưa có phép đo xác nhận.

Kết quả ba mục: [báo cáo 03/11/2026](reports/first_three_checkpoints_20261103.md).
Webcam đạt 24,61 FPS; processing P95 63,20 ms. Đã kiểm camera release, reset và
đổi GRU/DANN. Sensor frame drops/tuổi frame chưa đo được bằng API đang dùng;
mốc capture-to-result vẫn chưa xác nhận và chưa có nhãn để tính báo sai/giờ.

- [ ] **Theo dõi tiếp benchmark:** phân tích score đầu ở 149,62 giây, bổ sung đo
  tuổi frame/capture-to-result và phiên có ground truth để tính báo sai/giờ.

## Ưu tiên 2 — bổ sung bằng chứng còn thiếu của tuần 5

- [ ] **Khôi phục 44 model đối chứng cũ** cùng config, protocol, prediction và
  feature phù hợp; kiểm hash rồi so sánh cùng thời điểm, tách task binary/3 lớp.
  Đầu ra: bảng đối chứng có CI theo người và coverage. Nếu không còn bản sao,
  ghi rõ mục chưa khôi phục; thí nghiệm chạy lại mang tên/phiên bản mới.
- [ ] **Khôi phục bộ low-light.** Cần video validation nguồn và bộ 140 NPZ cũ,
  hoặc tái chạy 20 video ×7 điều kiện bằng script/config hiện có. Đầu ra: báo cáo
  accuracy, macro-F1, coverage và lý do từ chối theo điều kiện. Bộ RGB giảm sáng
  chỉ hỗ trợ đánh giá biến đổi ánh sáng, chưa xác nhận khả năng chạy ban đêm thật.
- [ ] **Khôi phục/tải DROZY và audit intake.** Đối chiếu 36 phiên/14 người được
  báo cáo; kiểm tính toàn vẹn, danh mục video/PSG/KSS/PVT và file thời gian.
  Đầu ra: manifest/audit chỉ rõ file đủ, thiếu và chưa xác minh.
- [ ] **Cập nhật đối chiếu tuần 5 sau khi có artifact.** Gắn từng kết luận với
  checkpoint, prediction, hash hoặc báo cáo chạy lại; giữ riêng trạng thái phục
  hồi công việc và trạng thái chưa đạt mục tiêu chất lượng.

## Tuần 6 — dữ liệu, phân tích và cải thiện

- [ ] **Hoàn tất A2 review 40 clip.** Khôi phục/tạo lại video và biểu mẫu; người
  đánh giá xem theo protocol, ghi interval và uncertainty, phân xử bất đồng trước
  khi xuất nhãn phiên bản mới. Phụ thuộc video nguồn và người review.
- [ ] **Khóa các thí nghiệm cải tiến từ kết quả phân tích lỗi.** Chọn ablation có
  lý do, ghi config/seed/split/tiêu chí lựa chọn trước khi chạy. Dùng 30 run đã xác
  minh làm baseline; ưu tiên GRU theo paired CI hiện có. Đầu ra: kế hoạch chạy Kaggle.
- [ ] **Chạy và so sánh ablation trên Kaggle.** Báo đủ seed 42/43/44, fold, CI theo
  người, macro-F1 và coverage; quyết định bằng argmax theo protocol. Đầu ra: bảng
  kết quả tái lập được và quyết định giữ/loại từng cải tiến.
- [ ] **Xác minh đồng bộ DROZY.** Kiểm CamSync/interpIndices, người/phiên và sai số
  căn chỉnh camera–PSG. Chỉ triển khai teacher EEG/student sau khi đồng bộ và nhãn
  đã được kiểm chứng; đánh giá teacher với tách người phù hợp.
- [ ] **Chuẩn bị cohort độc lập ngày/đêm.** Chốt protocol, nguồn dữ liệu, nhãn và
  người development/test trước khi phát triển trên cohort mới. Đầu ra: tập test
  riêng có thể khóa để nghiệm thu ứng viên.

## Tuần 7 — chốt ứng viên và đánh giá toàn trình

- [ ] **Đóng băng ứng viên cuối từ development.** Chốt cấu hình/ensemble, xuất
  checkpoint camera-only, input contract, hash và lý do lựa chọn; checkpoint từng
  fold CV tiếp tục phục vụ đối chứng.
- [ ] **Hoàn thiện luồng model → cảnh báo → API/dashboard.** Kiểm chính sách cảnh
  báo, trạng thái cảm biến kém, reset, telemetry metadata và hành vi khi lỗi;
  đo độ trễ đến cảnh báo cùng số báo sai/giờ trên video liên tục.
- [ ] **Đánh giá ứng viên đã khóa trên cohort độc lập.** Báo riêng ngày/đêm,
  accuracy, macro-F1, recall/specificity, CI theo người và coverage. Kết luận
  đạt/chưa đạt >85% theo kết quả thực tế; thiếu cohort thì giữ trạng thái chưa nghiệm thu.
- [ ] **Đóng gói bàn giao.** Model card, checkpoint/config/dependencies, lệnh chạy,
  demo, báo cáo chất lượng/độ trễ, giới hạn sử dụng và manifest sao lưu.

EEG supervision đầy đủ, Agents và IoT/NIR phần cứng tuân theo các điều kiện trong
lộ trình gốc; lịch tuần 7 không tự xác nhận hoàn thành các giai đoạn này.

## Tài liệu đối chiếu

- [Báo cáo khôi phục tuần 5](reports/week5_recovery_20261103.md).
- [Protocol tuần 5](../configs/training/evaluation_protocol_v1.json).
- [Kế hoạch sau khôi phục](ROADMAP.md).
- [Lộ trình model gốc](14_model_roadmap.md).
- [Camera, EEG và thiếu sáng](10_camera_eeg_night_plan.md).
- [Protocol A2](16_camera_annotation_protocol.md).
