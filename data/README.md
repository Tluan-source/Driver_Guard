# data/ (không commit)

```
data/
  raw/                 video gốc tải về (UTA-RLDD, YawDD, DMD, State Farm) — KHÔNG commit, KHÔNG chia sẻ lại
  features/<dataset>/  <video_id>.parquet — FrameSignals trích từ Kaggle notebook
  labels/              nhãn khoảng sự kiện (CSV) cho tập vàng tự quay
  golden_set/          video tự quay — chỉ lưu trên máy được mã hoá, xoá theo thời hạn trong đơn đồng ý
  runtime/             SQLite sự kiện khi chạy `driverguard serve/run --db`
```

Chi tiết nguồn, giấy phép và cách chia tập: `docs/02_data_plan.md`.
