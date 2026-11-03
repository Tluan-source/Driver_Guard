# Ba mục ưu tiên — 03/11/2026

## Git và bản sao

Đã commit source khôi phục tại `7f9d7b4` trên nhánh `recovery/week7`.
Push cần xác thực GitHub trên máy. Thư mục sao lưu Google Drive:
https://drive.google.com/drive/u/0/folders/14SteOCoccWKFaY7lUN7HyanQ1dEuWcwB
Trạng thái upload đang được kiểm chứng; chưa coi tạo thư mục là sao lưu hoàn tất.

## Phân tích lỗi 30 run CV

Chạy `python -m scripts.evaluation.analyze_week5_errors`.
Đầu ra: `outputs/week5_error_analysis/` gồm bảng theo run, người, fold, seed,
lớp, khoảng chất lượng; lý do từ chối, danh sách lỗi ensemble và summary JSON.
121.050 dòng là tổng qua 30 run, có lặp thời điểm giữa seed; mỗi ensemble có
20.175 thời điểm dự kiến, 19.812 được nhận. Test gốc không được mở.

| Ensemble | TN | FP | FN | TP | Recall drowsy | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 8020 | 2046 | 2704 | 7042 | 72,26% | 79,67% |
| DANN | 7910 | 2156 | 2713 | 7033 | 72,16% | 78,58% |

Ưu tiên kiểm tra người 28 (GRU accuracy 20,81%) và 19 (32,56%): coverage đều
100% nên chất lượng cảm biến không giải thích toàn bộ lỗi. Người 44 có accuracy
40,26%, coverage 79,85%, cần kiểm cả tín hiệu và nhãn. Đây là thứ tự review;
chưa đủ bằng chứng để kết luận nhãn sai hoặc nguyên nhân sinh lý.

Mỗi mode từ chối 363 thời điểm duy nhất: 163 current_frame_invalid,
171 timestamp_gap và 29 insufficient_quality. Bảng theo từng run đếm cả 3 seed.
Nhóm valid_fraction >95% vẫn chỉ khoảng 75% accuracy qua các seed: ưu tiên
review biểu hiện theo thời gian và khác biệt giữa người trước khi thay quality gate.
Nhóm chất lượng thấp thiên về lớp alert; accuracy cao riêng nhóm đó không có
nghĩa recall drowsy tốt. Không suy ra false alerts/giờ từ số cửa sổ FP.

## Webcam thật

Chạy `python -m scripts.evaluation.benchmark_week5_webcam --seconds 150`.
USB2.0 HD UVC Webcam, CPU, checkpoint GRU fold1 seed42 (đối chứng development).

- 150,31 giây, 3699 frame, **24,61 FPS**.
- Processing P50 **28,97 ms**, P95 **63,20 ms**.
- Face-valid **97,27%**; 11 frame có score, score đầu ở **149,62 giây**.
- Không có khoảng cách timestamp đầu vào >300 ms.
- Không lưu ảnh/video. Log metadata ở `outputs/week5_webcam_benchmark/`.

Độ trễ trên đo từ sau OpenCV read đến kết quả, chưa bao gồm thời gian phơi sáng
và tuổi frame trong driver. Vì vậy chưa xác nhận mục tiêu capture-to-result
P95 ≤100 ms. FPS đo đạt mốc thiết kế ≥15 FPS trong phiên này.
Score đầu muộn hơn ngữ cảnh danh nghĩa 120 giây cần phân tích tiếp chuỗi bin
10 Hz và reset của buffer; không coi warmup thực tế đã đạt 120 giây.
Chưa đo frame drops ở sensor, mất camera vật lý, chuyển model hoặc báo sai/giờ
có ground truth; các kiểm tra reset/mất mặt mô phỏng trước đó là bằng chứng riêng.
