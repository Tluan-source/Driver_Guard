# DriverGuard

Hệ thống giám sát tài xế (DMS) phát hiện **buồn ngủ và mất tập trung**, xử lý **on-device**, chỉ phát ra
**metadata** (không bao giờ gửi ảnh khuôn mặt), quyết định cảnh báo bằng **máy trạng thái tất định**.

```
camera (RAM) → MediaPipe → FrameSignals → hiệu chỉnh theo tài xế → temporal → state machine → cảnh báo
                                                                                        └→ FastAPI/WebSocket → dashboard
```

## Trạng thái skeleton v0.1

| Phần | Trạng thái |
|---|---|
| Perception (EAR, MAR, blendshape, head pose, chất lượng mặt, điện thoại) | ✅ viết xong, test bằng landmarker giả — **cần chạy thử với webcam thật** |
| Temporal (PERCLOS-proxy, blink, ngáp, nhìn lệch, gật đầu, điện thoại) | ✅ + test |
| Hiệu chỉnh theo tài xế (Hướng 1) — `adaptive` vs `fixed` | ✅ bản baseline + test tổng hợp |
| Risk engine (hysteresis, dwell, cooldown, SENSOR_DEGRADED, ngữ cảnh, CAN giả lập) | ✅ + test theo ma trận kịch bản |
| Trip memory (fatigue index / phút, xu hướng Theil–Sen) | ✅ + test |
| FastAPI 2 vai trò, WebSocket, HITL `/config` có audit, phản hồi cảnh báo | ✅ viết xong + test (chạy khi có `fastapi`) |
| React dashboard 1 màn hình | ✅ đã render thử |
| Evaluation: event matching, false alerts/giờ, split theo người, kappa, operating curve, UTA-RLDD | ✅ + test |
| Kaggle notebook trích feature | ✅ đã chạy thử luồng (với landmarker giả) |
| Golden vectors cho bản port Android | ✅ `tests/golden/` |
| Docker | ✅ Dockerfile + compose (chưa build thử) |
| Android app | ⏳ GĐ1 — xem `android/README.md` |
| GRU/TCN (Hướng 2), học trọng số (Hướng 4), generalization gap (Hướng 3) | ⏳ GĐ2–3 |

## Bắt đầu nhanh (Windows, Python 3.11)

### Mô hình EEG đã huấn luyện

Pipeline mô hình độc lập nằm trong `driverguard/learning/`: EEG CNT → công suất phổ 30 kênh ×
5 dải tần → mô hình → score và metrics. Đã train trên 12 người của bộ Driver Fatigue EEG,
chia theo người 8/2/2. Checkpoint được chọn trên validation là Logistic Regression với
công suất tương đối; trên test AUROC **0,6464**, balanced accuracy **0,5000**.
Đây là checkpoint nghiên cứu dùng được để suy luận/đo metrics; ngưỡng hiện tại chưa đủ tin cậy
để cảnh báo thực tế. Nhánh EEG chưa được hợp nhất với camera hoặc nối vào risk engine.

```powershell
.venv\Scripts\python.exe -m driverguard.learning.cli predict --checkpoint models/eeg_vigilance_v2/model.pt --features data/features/fatigue_eeg/subject_03_normal.npz --out outputs/eeg_predictions.csv
```

Checkpoint và dữ liệu đã có trên máy làm việc, được Git ignore. Máy khác cần tải/train lại.
Xem [hướng dẫn mô hình](docs/05_model_usage.md), [kết quả thực nghiệm](docs/06_model_results.md)
và [kế hoạch nghiên cứu](docs/04_model_research_plan.md). Agents và IoT chưa triển khai.

### Camera Và Hệ Thống

```powershell
py -3.11 -m venv .venv ; .venv\Scripts\activate
pip install -e ".[api,eval,dev]"
python scripts\download_models.py
pytest

driverguard simulate --scenario drowsy          # không cần camera/model
driverguard run --source 0 --show               # webcam + cửa sổ xem trước cục bộ
driverguard serve --source synthetic:demo       # API :8000  (docs: /docs)
cd dashboard ; npm install ; npm run dev        # dashboard :5173 — manager/manager, driver01/driver01
```

## Cấu trúc

```
driverguard/        mã nguồn Python (xem docs/01_architecture.md)
  capture/ perception/ calibration/ temporal/ risk/ trip/ vehicle/ alerts/ storage/ privacy/ api/ evaluation/ sim/
configs/            default.yaml (mọi ngưỡng = seed, tune trên validation) · users.example.yaml
dashboard/          React + Vite + TypeScript
kaggle/             notebook trích feature + hướng dẫn
scripts/            download_models.py · make_golden.py · py_to_ipynb.py
tests/              pytest + golden vectors
docker/  android/  data/  models/
docs/               00 phạm vi · 01 kiến trúc · 02 dữ liệu · 03 việc cần làm · templates/
```

## Tài liệu

- [`docs/00_scope_mapping.md`](docs/00_scope_mapping.md) — ánh xạ đề bài → MVP / sau / loại (mang đi GVHD duyệt)
- [`docs/01_architecture.md`](docs/01_architecture.md) — luồng dữ liệu, luật quyết định, API, kiểm thử privacy
- [`docs/02_data_plan.md`](docs/02_data_plan.md) — dataset, giấy phép, chia tập, tập vàng tự quay
- [`docs/03_viec_can_lam.md`](docs/03_viec_can_lam.md) — checklist: cài đặt, tải dữ liệu, Kaggle, Giai đoạn 0
- `docs/DriverGuard_*.docx` — tài liệu nghiên cứu gốc và nhận xét phản biện

## Nguyên tắc không được phá

1. Không endpoint/đầu ra nào chứa ảnh; mọi payload qua `privacy.assert_metadata_only`.
2. Mất mặt/camera = `SENSOR_DEGRADED`, không bao giờ là `NORMAL` hay "buồn ngủ".
3. Mọi timer dùng timestamp, không dùng số frame.
4. LLM/SLM (nếu có) chỉ giải thích, không quyết định.
5. Chia dữ liệu theo người; chọn ngưỡng trên validation; chạy test một lần.
6. Kết quả trên dữ liệu tổng hợp (`sim/`) chỉ là kiểm tra logic — không báo cáo như kết quả thực nghiệm.
