# 00 — Bảng ánh xạ phạm vi: Summary.png → MVP / Giai đoạn sau / Ngoài phạm vi

> Trạng thái: **BẢN ĐỀ XUẤT — chờ GVHD duyệt** (Quyết định #1, GVHD mục 1 & 7.2).
> Nguyên tắc: mỗi yêu cầu của đề bài rơi vào đúng một ô — **(a) MVP**, **(b) giai đoạn sau kèm mốc**, hoặc
> **(c) ngoài phạm vi kèm lý do**. Không có ô "im lặng bỏ qua".
>
> Mốc tuần theo kế hoạch 1 học kỳ của GVHD (mục 7.3): GĐ0 tuần 1–2 · GĐ1 tuần 3–6 · GĐ2 tuần 7–10 ·
> GĐ3 tuần 10–13 · GĐ4 tuần 13–15 · GĐ5 tuần 15–16.

## Thực trạng / Vấn đề / Ràng buộc

| # | Yêu cầu trong đề bài | Ô | Xử lý & lý do | Ở đâu trong code |
|---|---|---|---|---|
| 1 | Camera **RGB** | (a) | Camera trước điện thoại / webcam | `capture/source.py` |
| 2 | Camera **hồng ngoại** (IR) | (c) cho MVP · (b) nếu mua được camera IR | Không có phần cứng IR; DMD 2026 đã gỡ dữ liệu IR → không có dữ liệu để phát triển. Kiến trúc không phụ thuộc loại camera (chỉ cần landmark) | — |
| 3 | Nhắm mắt kéo dài (PERCLOS) | (a) | Đổi tên thành **PERCLOS-proxy** (ước lượng từ EAR, không phải PERCLOS đo bằng IR) — GVHD 9.1 | `temporal/trackers.py` `EyeStateTracker` |
| 4 | Ngáp | (a) | Chỉ là **bằng chứng yếu** → tối đa CAUTION (GVHD 9.2) | `YawnTracker` |
| 5 | Gục đầu | (a) | Phát hiện gật đầu nhanh (pitch giảm đột ngột) | `HeadTracker` (nod) |
| 6 | Quay đi khỏi đường | (a) | Tư thế đầu **so với tư thế trung tính đã hiệu chỉnh**; liếc gương < 2,5 s không cảnh báo | `HeadTracker` |
| 7 | Cầm điện thoại | (a) | MediaPipe Object Detector (Apache-2.0) + gần mặt + kéo dài ≥ 1,5 s. Bàn tay (hand landmarks) → (b) GĐ1 | `perception/phone.py`, `PhoneTracker` |
| 8 | Agent lập luận rủi ro theo **ngữ cảnh** (tốc độ, thời gian lái liên tục) | (a) | Làm **tất định**: ngữ cảnh đổi độ nhạy ngưỡng, xe dừng → không cảnh báo điện thoại/nhắm mắt, lái ≥ 4 h → CAUTION. Không để LLM quyết định | `risk/context.py` |
| 9 | Cảnh báo phân cấp: âm thanh | (a) | NORMAL / CAUTION / WARNING / CRITICAL + SENSOR_DEGRADED | `risk/state_machine.py`, `alerts/` |
| 10 | …rung ghế | (b) GĐ1 (rung điện thoại) · (c) rung ghế | Rung ghế cần phần cứng xe; điện thoại có rung sẵn | `android/README.md` |
| 11 | …gợi ý nghỉ | (a) | Thông điệp tiếng Việt theo mức + lý do | `alerts/explain.py` |
| 12 | **Memory xu hướng mệt mỏi cả chuyến** | (a) | GVHD đánh giá là phần giá trị nhất → đưa vào MVP: fatigue index theo phút + độ dốc Theil–Sen → `FATIGUE_TREND_RISING` | `trip/memory.py` |
| 13 | On-device, không lưu/gửi ảnh mặt | (a) | Không có endpoint ảnh; `assert_metadata_only` chặn mọi payload; test bắt gói tin ở GĐ3 | `privacy/guard.py`, `api/app.py` |
| 14 | Chống báo động giả | (a) | Persistence, hysteresis, cooldown, **hiệu chỉnh theo tài xế** (Hướng 1) | `calibration/`, `risk/` |
| 15 | Realtime ≥ 15 FPS trên edge | (b) đo ở GĐ1–GĐ3 | Chỉ công nhận số đo **trên Android**; laptop không phải nghiệm thu | `evaluation/runtime_bench.py` |
| 16 | AI chỉ cảnh báo, không can thiệp lái | (a) | Không có đầu ra điều khiển xe (by design) | — |
| 17 | Hoạt động ban đêm | (c) cho RGB trong cabin tối · (a) đô thị đủ sáng | Không IR thì landmark không tin cậy trong tối → hệ thống báo SENSOR_DEGRADED thay vì đoán | `perception/quality.py` |
| 18 | Người đeo kính | (b) GĐ3 | Kính cận: đưa vào ma trận robustness. Kính râm: → SENSOR_DEGRADED | — |

## Kỹ thuật

| # | Yêu cầu | Ô | Xử lý & lý do | Ở đâu |
|---|---|---|---|---|
| 19 | Face/landmark nhẹ (MediaPipe/BlazeFace) | (a) | MediaPipe Face Landmarker (478 điểm, 52 blendshape, ma trận tư thế) | `perception/face.py` |
| 20 | Eye-state / head-pose model (MobileNet) | (a) hình học · (b) CNN nếu cần | EAR + blendshape + ma trận tư thế trước; CNN mắt chỉ thêm nếu EAR không đủ (đo được) | `perception/geometry.py`, `head_pose.py` |
| 21 | YOLOv8-nano | (c) thay bằng EfficientDet-Lite0 | YOLOv8 (Ultralytics) là **AGPL-3.0**; EfficientDet-Lite0 của MediaPipe là Apache-2.0 và chạy native trên Android | `perception/phone.py` |
| 22 | ONNX Runtime / TFLite lượng tử hoá | (b) GĐ3 | MediaPipe đã dùng TFLite; lượng tử hoá INT8 + đo lại accuracy khi có model riêng | — |
| 23 | Luật PERCLOS + temporal smoothing | (a) | Mọi timer theo timestamp, không theo số frame | `temporal/primitives.py` |
| 24 | SLM lập luận ngữ cảnh (tuỳ chọn) | (b) GĐ3, chỉ **giải thích** | Giao diện một chiều `(level, reasons) -> text`; không đổi quyết định | `alerts/explain.py` |
| 25 | Mô phỏng tốc độ qua **CAN giả lập** | (a) | Mã hoá/giải mã frame CAN 0x3E9; các profile mixed/highway/urban/parked | `vehicle/can_sim.py` |
| 26 | Backend FastAPI | (a) | REST + WebSocket, chỉ metadata | `api/app.py` |
| 27 | React dashboard + mô phỏng màn hình cảnh báo | (a) dashboard · (b) GĐ1 màn hình trong xe | Dashboard 1 màn hình; màn hình cảnh báo trong xe = app Android | `dashboard/` |
| 28 | Dữ liệu video DMS công khai | (a) | UTA-RLDD, DMD, YawDD, State Farm + **tập vàng tự quay** | `docs/02_data_plan.md` |
| 29 | Mô phỏng Jetson | (b) GĐ3 | Docker ARM64 (kiểm tra chức năng, không phải hiệu năng); Jetson thật nếu có ngân sách | `docker-compose.yml` |
| 30 | Docker | (a) | Backend + dashboard một image | `docker/Dockerfile` |

## Cơ bản / Nâng cao

| # | Yêu cầu | Ô | Xử lý & lý do | Ở đâu |
|---|---|---|---|---|
| 31 | Phát hiện trên video mẫu, phân cấp cảnh báo | (a) | `driverguard run --source video.mp4` | `cli.py` |
| 32 | Đăng nhập ≥ 2 vai trò (tài xế / fleet manager) | (a) | Tài xế chỉ thấy dữ liệu của mình; quản lý thấy metadata mọi tài xế, **không bao giờ thấy ảnh** | `api/auth.py` |
| 33 | HITL hiệu chỉnh ngưỡng | (a) | 9 khoá được phép chỉnh (có biên), chỉ fleet_manager, ghi `config_audit`; tài xế phản hồi Đúng/Sai cho từng cảnh báo | `config.py` `TUNABLE_KEYS`, `ConfigPanel.tsx` |
| 34 | Dashboard nhật ký sự kiện | (a) | Bảng sự kiện + chuyến hiện tại + biểu đồ 5 phút | `dashboard/src` |
| 35 | Metric precision / recall / false positive | (a) | Báo cáo **đường cong recall theo false alerts/giờ**, chọn điểm vận hành trên validation (GVHD 8.3) | `evaluation/` |
| 36 | Lượng tử hoá để tối ưu FPS/độ trễ, chạy offline | (b) GĐ3 · offline (a) | Hệ thống không cần mạng để phát hiện | — |
| 37 | **Hiệu chỉnh cá nhân hoá theo tài xế** | (a) — **đóng góp AI chính** | Hướng 1 của GVHD: baseline EAR thích ứng, so sánh với ngưỡng cố định | `calibration/baseline.py` |
| 38 | Eval nhiều điều kiện ánh sáng | (b) GĐ3 | Ma trận 19 kịch bản robustness | `docs/02_data_plan.md` |
| 39 | Giám sát đội xe nhiều thiết bị | (b) GĐ4 | API đã theo `driver_id`/`trip_id`; gom nhiều thiết bị khi chạy thử | `api/app.py` `/trips` |
| 40 | OTA cập nhật ngưỡng/model | (c) | Cần hạ tầng ký số + phân phối; HITL cục bộ đã đáp ứng nhu cầu chỉnh ngưỡng trong phạm vi đồ án | — |
| 41 | Guardrails chống lạm dụng dữ liệu khuôn mặt | (a) | Không nhận diện danh tính, không embedding, `driver_id` giả danh, phân quyền, xoá theo `retention_days`, debug recording tắt | `privacy/`, `storage/` |

## Sáu quyết định cần chốt (GVHD 7.2) — trạng thái trong repo

| # | Quyết định | Trạng thái |
|---|---|---|
| 1 | Bảng ánh xạ này | Đề xuất — **mang đi GVHD duyệt tuần 1** |
| 2 | Nền tảng: Android là đích, laptop để phát triển/đánh giá | Đã chốt với nhóm: **Python core trước**, port Android ở GĐ1 (kiểm chứng bằng golden vectors) |
| 3 | Chiến lược nhãn: DMD chủ lực + tập vàng tự quay 90–120 phút + 16–24 giờ gán nhãn | Đề xuất trong `02_data_plan.md` — cần phân công người |
| 4 | Đóng góp AI: Hướng 1 chính; Hướng 2 (GRU/TCN vs rule) & 3 (generalization gap) phụ | Khung đã sẵn: `calibration.method: adaptive|fixed`, operating curves, feature replay |
| 5 | Persona kép: người mua = quản lý đội xe; người dùng = tài xế (có thể đối kháng) | Phản ánh trong 2 vai trò + chỉ số **tỷ lệ degraded** (dấu hiệu che camera) |
| 6 | Quy trình đồng ý cho video tự quay | Mẫu: `templates/mau_don_dong_y.md` — **phải ký trước buổi quay đầu tiên** |
