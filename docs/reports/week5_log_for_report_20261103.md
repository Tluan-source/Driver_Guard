# Log tuần 5 để điền báo cáo

Cập nhật ngày 03/11/2026. Nội dung tổng hợp công việc khôi phục và kiểm chứng
hiện tại. Metrics CV xuất phát từ các run Kaggle đã có, được tái tính và xác minh
trên máy mới; ngày cập nhật log không phải ngày huấn luyện các checkpoint này.

## Nội dung công việc đã thực hiện

1. Khôi phục source/config/notebook tuần 5, 30 checkpoint GRU/GRU+DANN và 151
   file feature NPZ của 50 người development. Kiểm tính toàn vẹn ZIP, hash artifact
   và 138 file source được ghi trong provenance huấn luyện.
2. Kiểm chứng protocol subject-CV 5 fold, mỗi fold 32 người train, 8 người inner
   validation, 10 người outer development; seed 42/43/44. Kiểm split tách người
   và loại 10 người UTA test gốc khỏi tập development.
3. Tái tính kết quả ensemble ba seed, confusion matrix, macro-F1, coverage và CI
   bootstrap theo người. Dùng task alert/drowsy, cửa sổ 120 giây, 10 Hz, stride
   2,5 giây, quyết định argmax theo protocol đã khóa.
4. Replay CPU cả 30 checkpoint trên 960 cửa sổ inner validation lấy mẫu; kiểm
   thêm một clip đầy đủ qua API Model Lab với 204 thời điểm đối chiếu CSV Kaggle.
5. Khôi phục dashboard và Model Lab: chọn checkpoint, replay feature, mô phỏng
   và kiểm tra runtime. Loại checkpoint từng fold CV khỏi lựa chọn model cuối.
   Kiểm tra warmup, mất mặt, reset và chặn score khi không đánh giá được.
6. Phân tích 121.050 dòng dự đoán của 30 run theo người, fold, seed, lớp, nhóm
   chất lượng và lý do từ chối; xuất CSV tổng hợp và danh sách lỗi ensemble.
7. Benchmark webcam thật trên CPU trong 150 giây; kiểm bổ sung camera ngừng đọc
   sau khi release, reset runtime và chuyển giữa checkpoint GRU/DANN.
8. Chạy bộ kiểm thử phục hồi: 307 test đạt; hai test mới cho phân tích lỗi đạt
   trong lần chạy riêng. Build dashboard thành công, kiểm lint các script mới đạt.
9. Commit/push source, test, docs và kết quả tổng hợp lên nhánh `recovery/week7`;
   sao lưu hai ZIP model/data lên Google Drive, lưu checksum trong Git.

## Kết quả đạt được

| Chỉ tiêu development CV | GRU | GRU+DANN |
|---|---:|---:|
| Window accuracy | 76,0246% | 75,4240% |
| Macro-F1 | 75,9661% | 75,3757% |
| Coverage | 98,2007% | 98,2007% |
| Recall drowsy | 72,2553% | 72,1629% |
| Specificity | 79,6742% | 78,5814% |
| Cửa sổ nhận /dự kiến | 19.812 /20.175 | 19.812 /20.175 |

Paired CI95% cho chênh lệch accuracy DANN − GRU là [-1,3889; +0,1065] điểm
phần trăm, chứa 0. Giữ GRU làm hướng baseline theo quy tắc đã khai báo.
Các số liệu trên thuộc development CV; nghiệm thu độc lập chưa thực hiện.

- Replay CPU: sai lệch xác suất lớn nhất 0,00000453, trong tolerance 0,00001.
- Webcam: 3.699 frame/150,31 giây, 24,61 FPS; processing P50 28,97 ms,
  P95 63,20 ms; face-valid 97,27%. Chỉ lưu metadata, ảnh/video không được lưu.
- Score đầu trong phiên webcam xuất hiện ở giây 149,62; cần điều tra thêm warmup
  thực tế so với ngữ cảnh danh nghĩa 120 giây.
- Phân tích lỗi: ưu tiên review người 28 và 19 do accuracy thấp dù coverage 100%;
  chưa kết luận nguyên nhân hoặc nhãn sai khi chưa review video.

## Khó khăn và phần chưa hoàn thành

- Accuracy development hiện tại chưa đạt mục tiêu >85%.
- Chưa khôi phục đủ 44 checkpoint đối chứng cũ, bộ low-light 140 NPZ/video nguồn
  và dữ liệu DROZY để tái kiểm chứng toàn bộ nội dung tuần 5.
- Chưa đo capture-to-result đầy đủ, tuổi frame/frame drops ở sensor và báo sai/giờ
  trên phiên có ground truth. Processing latency chưa thay thế các chỉ tiêu này.
- Chưa có đánh giá cohort độc lập ngày/đêm; EEG teacher–student và căn chỉnh
  camera–PSG vẫn cần dữ liệu đồng bộ đã kiểm chứng.

## Công việc tiếp theo

1. Phân tích warmup webcam 149,62 giây và bổ sung đo độ trễ toàn trình.
2. Review các nhóm lỗi theo người/lớp, hoàn tất A2 40 clip khi có video và người
   đánh giá; lập ablation cải tiến trên Kaggle.
3. Khôi phục các model đối chứng, dữ liệu low-light và DROZY; kiểm đồng bộ trước
   thí nghiệm EEG supervision.
4. Chuẩn bị cohort độc lập, đóng băng ứng viên rồi đánh giá ngày/đêm và cảnh báo.

## Minh chứng

- [Code và báo cáo trên GitHub](https://github.com/Tluan-source/Driver_Guard/tree/recovery/week7).
- [Bản sao model/data trên Google Drive](https://drive.google.com/drive/u/0/folders/14SteOCoccWKFaY7lUN7HyanQ1dEuWcwB).
- [Báo cáo khôi phục](week5_recovery_20261103.md).
- [Phân tích lỗi và benchmark webcam](first_three_checkpoints_20261103.md).
- CSV/JSON tổng hợp: `docs/results/week5_error_analysis/`,
  `docs/results/week5_webcam_benchmark.json`, `docs/results/week5_webcam_controls.json`.
