# Lộ trình sau khôi phục

Cập nhật 03/11/2026. Bảy phần kỹ thuật trong `configs/project.json` khác với
bảy tuần của sheet tiến độ. Báo cáo tuần 5 đối chiếu tại
[báo cáo khôi phục](reports/week5_recovery_20261103.md).

## Chốt phần tuần 5

Đã có source, protocol khóa, 30 checkpoint, 151 NPZ, báo cáo CV, replay CPU,
runtime 120 giây và UI. Chưa đủ artifact để kiểm chứng lại so sánh 44 model cũ,
low-light 20 video ×7 điều kiện, và intake DROZY 36 phiên. Cần khôi phục bản sao
các artifact này hoặc chạy lại từ dữ liệu gốc rồi ghi rõ đó là lần chạy mới.

Mục tiêu accuracy >85% chưa đạt; kết quả development khoảng 75–76% là kết quả
thực nghiệm cần giữ nguyên. Không dùng checkpoint từng fold làm model cuối.

## Tuần 6: phân tích lỗi và dữ liệu

Theo các việc còn lại trong sheet tuần 5 và [kế hoạch camera/EEG](10_camera_eeg_night_plan.md):

1. Phân tích lỗi theo người, lớp và chất lượng/coverage trên prediction development;
   xem các trường hợp mất mặt, nhắm mắt và góc đầu, không chọn ngưỡng bằng test.
2. Hoàn tất review A2 40 clip bằng người đánh giá, giữ nguồn nhãn và timestamp.
3. Khôi phục low-light RGB và báo cáo accuracy cùng coverage theo điều kiện;
   giữ riêng kết quả RGB giả lập tối với dữ liệu đêm thật.
4. Audit DROZY đúng người/phiên; kiểm CamSync/interpIndices trước mọi căn chỉnh
   camera–PSG. Chỉ chạy teacher EEG sau khi có chứng cứ đồng bộ.
5. Khóa các ablation cải tiến, chạy tác vụ nặng trên Kaggle với seed/protocol đã ghi.

Đầu ra: báo cáo lỗi có thể tái lập, dữ liệu/nhãn được kiểm định, bảng ablation
development cùng CI và coverage. Không mặc định sẽ đạt 85%.

## Tuần 7: ứng viên, ứng dụng và đánh giá

Đây là kế hoạch triển khai suy ra từ tài liệu dự án; sheet đã đọc không có tab tuần 7.

1. Chốt cấu hình/ensemble dựa trên development; xuất checkpoint camera-only cùng
   input contract, hash và quyết định lựa chọn.
2. Kiểm thử video liên tục/webcam local: warmup, mất cảm biến, reset, độ trễ và
   false alerts/giờ; kiểm dashboard và luồng cảnh báo.
3. Đánh giá cohort độc lập ngày/đêm sau khi khóa ứng viên và protocol. Nếu chưa có
   cohort, ghi rõ chưa nghiệm thu; không đổi tên CV thành test độc lập.
4. Đóng gói hướng dẫn chạy, artifact manifest, kết quả và giới hạn. EEG supervision,
   Agent và IoT chỉ triển khai khi có dữ liệu và hạ tầng đúng điều kiện trong docs.

## Bảy phần kỹ thuật

| Phần | Hiện tại | Điều kiện tiếp theo |
|---|---|---|
| 01. Môi trường/cấu trúc | Đã khôi phục local | Giữ kiểm thử và hướng dẫn cài tái lập |
| 02. Dữ liệu/nhãn | 151 NPZ development; còn thiếu raw/A2/DROZY | Khôi phục và kiểm định dữ liệu |
| 03. Feature camera | Dùng lại feature gốc khớp hash | Khôi phục low-light và dữ liệu đối chứng |
| 04. GRU/GRU+DANN | 30 run CV, chưa >85% | Phân tích lỗi, ablation và chọn cấu hình |
| 05. EEG supervision | Có code, thiếu dữ liệu đồng bộ | Audit/alignment trước teacher–student |
| 06. Đánh giá/inference | Replay và runtime local đã kiểm tra | Cohort độc lập và benchmark webcam |
| 07. Ứng dụng/Edge/IoT | Dashboard và Model Lab chạy local | Kiểm thử tích hợp và phần cứng thật |
