# data/ (không commit)

Khôi phục ngày 03/11/2026: `features/week5_development/` có 151 NPZ của 50 người
development (train/validation gốc). `recovery_week5/` giữ nguyên source/feature từ
ZIP Kaggle và manifest hash. Không có 10 người UTA test gốc, video raw, bộ low-light
140 NPZ hoặc dữ liệu DROZY trong bản khôi phục này.

Manifest feature local chỉ đổi đường dẫn; byte NPZ giữ nguyên. Nhãn gốc ba lớp
alert/low_vigilance/drowsy; trainer `alert_drowsy` chỉ giữ alert và drowsy.

```
data/
  raw/                 video gốc tải về (UTA-RLDD, YawDD, DMD, State Farm) — KHÔNG commit, KHÔNG chia sẻ lại
  features/<dataset>/  <clip_id>.npz + manifest.json — đặc trưng camera, timestamp, validity và provenance
  labels/              nhãn khoảng sự kiện (CSV) cho tập vàng tự quay
  golden_set/          video tự quay — chỉ lưu trên máy được mã hoá, xoá theo thời hạn trong đơn đồng ý
  runtime/             SQLite sự kiện khi chạy `driverguard serve/run --db`
```

Chi tiết nguồn, giấy phép và cách chia tập: `docs/02_data_plan.md`.
