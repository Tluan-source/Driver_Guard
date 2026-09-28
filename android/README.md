# android/ — kế hoạch port (Giai đoạn 1, tuần 3–6)

Android là **nền tảng đích** (GVHD mục 7.1). Laptop chỉ để phát triển và đánh giá; mọi con số FPS/độ trễ
nghiệm thu phải đo trên điện thoại.

## Kiến trúc port

```
CameraX (front camera, 640x480)
   -> MediaPipe Tasks Android: FaceLandmarker (LIVE_STREAM) + ObjectDetector (efficientdet_lite0)
   -> FrameSignals (Kotlin data class, cùng tên trường với driverguard/schemas.py)
   -> Engine (Kotlin port của driverguard/engine.py + temporal/ + calibration/ + risk/ + trip/)
   -> TickOutput -> cảnh báo (âm thanh + rung + TTS tiếng Việt) + lưu Room/SQLite (metadata)
   -> (tuỳ chọn) gửi metadata tới FastAPI qua WebSocket/HTTP — KHÔNG bao giờ gửi frame
```

## Cách đảm bảo port đúng: golden test vectors

`tests/golden/*.jsonl.gz` chứa đầu vào FrameSignals + đầu ra mong đợi (risk_level, reason_codes, alert,
closure_ms, perclos_proxy) do engine Python sinh ra (`python scripts/make_golden.py`).
Unit test Kotlin phải đọc cùng các file này và khớp **từng tick**. Khi đổi luật/ngưỡng mặc định:
regenerate golden bằng Python trước, rồi sửa Kotlin cho khớp.

## Việc cần quyết định sớm (ràng buộc sản phẩm)

- Tài xế dùng chính điện thoại để chạy app gọi xe / bản đồ → app phải chạy **foreground service**
  (camera ở nền) hoặc overlay nhỏ. Kiểm tra giới hạn camera nền trên Android 14+.
- Đo nhiệt: chạy 30–60 phút trong cabin, ghi FPS theo phút (tương đương `driverguard.evaluation.runtime_bench`).

## Việc chưa làm trong skeleton này

- [ ] Tạo project Android Studio (Kotlin, minSdk 26), module `engine` thuần Kotlin (không phụ thuộc Android) để unit-test golden.
- [ ] Port `temporal/primitives.py` → `trackers.py` → `calibration/baseline.py` → `risk/*` → `trip/memory.py`.
- [ ] CameraX + MediaPipe LIVE_STREAM; timestamp dùng `SystemClock.elapsedRealtime()`.
- [ ] Benchmark FPS/nhiệt trên máy thật.
