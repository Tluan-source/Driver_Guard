# Tài liệu DriverGuard

Điểm bắt đầu hiện tại: [khôi phục tuần 5 ngày 03/11/2026](reports/week5_recovery_20261103.md).
Các báo cáo trong `results/` và tài liệu dưới đây là bằng chứng lịch sử; việc có
metrics trong Git không có nghĩa checkpoint hoặc dữ liệu tương ứng đã được khôi phục.

- [Lộ trình và việc tiếp theo](ROADMAP.md).
- [Checklist công việc tiếp theo](NEXT_STEPS.md).
- [Phạm vi bài toán](00_scope_mapping.md), [kiến trúc](01_architecture.md), [dữ liệu](02_data_plan.md).
- [Lộ trình model](14_model_roadmap.md), [camera, EEG và low-light](10_camera_eeg_night_plan.md).
- [Cách dùng camera](11_camera_model_usage.md), [kết quả camera](12_camera_model_results.md).
- [Cách dùng EEG](05_model_usage.md), [kết quả EEG v3](09_model_v3_results.md).
- [Protocol review nhãn A2](16_camera_annotation_protocol.md), [camera development](15_camera_development_results.md).

Source Kaggle đã chuyển camera/EEG vào các package con. Các entry point phẳng
trong tài liệu cũ vẫn có alias tương thích. Lệnh hiện hành nằm trong README gốc
và `python -m driverguard.project tasks`.
