# DriverGuard

Giám sát buồn ngủ/mất tập trung từ camera và cảnh báo tại thiết bị. Hướng nghiên cứu:
camera → GRU/DANN → EEG supervision → model cuối chỉ dùng camera.

## Trạng thái sau khôi phục ngày 03/11/2026

Đã khôi phục source Kaggle tuần 5, **30 checkpoint CV** và **151 file feature của
50 người development**. Đã kiểm hash, tái tính metrics/CI, replay CPU cả 30 model
và kiểm tra runtime 120 giây. Chi tiết và bằng chứng ở
[báo cáo khôi phục tuần 5](docs/reports/week5_recovery_20261103.md).

| Ensemble 3 seed, 5 fold development | Window accuracy | Macro-F1 | Coverage |
|---|---:|---:|---:|
| GRU | 76,0246% | 75,9661% | 98,2007% |
| GRU+DANN | 75,4240% | 75,3757% | 98,2007% |

Đây là kết quả development, **chưa đạt mục tiêu accuracy >85%** và chưa phải test
độc lập. Còn thiếu artifact của 44 model đối chứng cũ, bộ low-light 140 NPZ/video
nguồn và dữ liệu DROZY. Vì vậy chưa xác nhận khôi phục đủ toàn bộ nội dung tuần 5.

## Chạy trên máy này

Mở PowerShell tại thư mục project. Môi trường đã cài ở `.venv` (Python 3.10).

```powershell
.\.venv\Scripts\python.exe -m driverguard.project status
.\.venv\Scripts\python.exe -m driverguard.project doctor

# Model Lab: chọn checkpoint, replay feature, mô phỏng hoặc mở webcam bằng nút trên UI
.\.venv\Scripts\python.exe -m driverguard debug-ui
# http://127.0.0.1:8010

# Dashboard: quản lý artifact, CV và tiến độ (không tự sinh telemetry)
.\.venv\Scripts\python.exe -m driverguard.api.project_server
# http://127.0.0.1:8000 — tài khoản demo manager / manager
```

Model Lab không cần build; source nằm ở `ui/test-console/`. Dashboard React nằm ở
`dashboard/`, build bằng `npm ci` rồi `npm run build` trong thư mục đó. API vẫn hỗ
trợ layout `ui/dashboard/dist` của source Kaggle. Tài khoản demo khai báo trong
`configs/users.example.yaml`; thông tin đăng nhập riêng dùng `configs/users.yaml`.

Để chạy camera bằng cửa sổ local:

```powershell
.\.venv\Scripts\python.exe -m driverguard.project run camera-live
```

Lệnh này mở camera và checkpoint CV để kiểm thử runtime. Checkpoint từng fold
không phải model nghiệm thu cuối. Model cần 120 giây ngữ cảnh liên tục; mất mặt
không được coi là trạng thái tỉnh táo. Nút reset xóa lịch sử và bắt đầu warmup lại.

## Kiểm tra lại kết quả

```powershell
.\.venv\Scripts\python.exe -m driverguard.project run week5-audit
.\.venv\Scripts\python.exe -m driverguard.project run week5-replay
.\.venv\Scripts\python.exe -m driverguard.project run week5-runtime
.\.venv\Scripts\python.exe -m driverguard.project run week5-summary
.\.venv\Scripts\python.exe -m pytest -q -o addopts=
```

Các báo cáo mới ghi vào `outputs/recovery_week5/`; giữ nguyên bản gốc trong
`models/recovered_week5/` và `data/recovery_week5/`. File `test_windows.csv` trong
30 run là **outer fold của development**, không phải 10 người UTA test gốc.
Không mở test gốc để chọn kiến trúc/ngưỡng. Protocol ở
`configs/training/evaluation_protocol_v1.json`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `driverguard/learning/camera/` | Feature, GRU/DANN, train và inference camera |
| `driverguard/learning/eeg/` | EEG preprocessing, teacher và evaluation |
| `driverguard/debug/`, `ui/test-console/` | Model Lab local; replay/live |
| `driverguard/api/`, `dashboard/` | API metadata và dashboard quản lý |
| `scripts/data/`, `scripts/evaluation/` | Chuẩn bị dữ liệu, CV, audit và benchmark |
| `kaggle/04_week5_subject_cv.py` và `.ipynb` | 30 run trên Kaggle T4 ×2 |
| `data/`, `models/`, `outputs/`, `runs/` | Dữ liệu và artifact local, được Git ignore |
| `docs/` | [Mục lục](docs/README.md), [lộ trình](docs/ROADMAP.md), báo cáo và tài liệu gốc |

## Cài trên máy khác

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api,eval,train,dev]" av
.\.venv\Scripts\python.exe -m scripts.data.download_models
```

Checkpoint và feature không nằm trong Git. Xem cách nhập hai ZIP trong
[báo cáo khôi phục](docs/reports/week5_recovery_20261103.md).

Chia tập theo người trước cắt cửa sổ; chọn epoch bằng inner validation, quyết định
bằng argmax không tune threshold. Không ghép EEG khác người/phiên với UTA.
API sản phẩm chỉ phát metadata; ảnh webcam trong console local chỉ xử lý trong RAM.
