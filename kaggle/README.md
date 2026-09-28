# Kaggle — những gì chạy trên Kaggle và vì sao

Phần nặng duy nhất cần GPU/CPU mạnh hơn laptop là **chạy MediaPipe trên hàng chục giờ video dataset**.
Ta chạy nó **một lần** trên Kaggle, lưu `FrameSignals` ra parquet, rồi mọi thí nghiệm sau (ngưỡng, hiệu chỉnh
theo tài xế, GRU/TCN, học trọng số) chạy lại trên laptop trong vài giây bằng `driverguard replay`.

| Notebook | Làm gì | Khi nào |
|---|---|---|
| `01_extract_features.ipynb` | Video → `<video_id>.parquet` (EAR, MAR, blendshape, head pose, quality) | Ngay khi có dataset |
| `02_train_temporal.ipynb` *(chưa có)* | GRU/TCN trên chuỗi feature, đối chứng với rule engine (Hướng 2) | Giai đoạn 2 (tuần 7–10) |

## Chạy `01_extract_features` — từng bước

1. **Đóng gói mã nguồn thành Kaggle dataset** (làm 1 lần, cập nhật khi code đổi):
   - Nén thư mục `driverguard/` thành `driverguard-src.zip` (giữ nguyên tên thư mục `driverguard/` ở gốc zip).
   - Kaggle → Datasets → New Dataset → upload zip → đặt tên `driverguard-src` → **Private**.
2. **Tạo notebook**: Kaggle → Code → New Notebook → File → Import Notebook → chọn `kaggle/01_extract_features.ipynb`.
3. **Add Input**:
   - `driverguard-src` (vừa tạo);
   - dataset video. UTA-RLDD có sẵn bản mirror cộng đồng: `rishab260/uta-reallife-drowsiness-dataset`
     (**kiểm tra cấu trúc thư mục Fold/Subject/0|5|10 và đủ 60 người** — nếu thiếu, dùng bản Google Drive chính thức).
4. **Settings**: Internet **ON** (cần để `pip install mediapipe` và tải `face_landmarker.task`), Accelerator: None.
5. Ô CONFIG: đặt `DATA_ROOT` đúng đường dẫn `/kaggle/input/...`, `FOLDS = [1]` (phiên đầu), giữ `TARGET_FPS = 15`.
6. **Save Version → Save & Run All (Commit)** để chạy nền (không cần giữ tab mở). Ước lượng: 1 fold UTA-RLDD
   (~6 giờ video) ≈ 1–2 giờ trên 4 CPU — con số thật sẽ nằm trong `run_<tag>.json`.
7. Xong → tab **Output** → tải `features_uta_rldd_fold1.zip` → giải nén vào `data/features/uta_rldd/`.
8. Lặp lại với `FOLDS = [2]`, `[3]`… (hoặc `[2, 3]` nếu phiên đầu chạy nhanh).
   Muốn chạy tiếp phần dở: attach output phiên trước làm input và đặt `PREVIOUS_OUTPUT`.

## Sau khi có feature (trên laptop)

```bash
# manifest.json do notebook ghi sẵn trong thư mục feature (fold/subject/label)
python -m driverguard.evaluation.uta_rldd evaluate --manifest data/features/uta_rldd/manifest.json --features data/features/uta_rldd
driverguard replay --features data/features/uta_rldd/01_10.parquet --jsonl runs/01_10.jsonl
```

> Không cần tải video về máy nếu chỉ làm thí nghiệm trên feature. Nếu có video gốc trên máy, có thể tự tạo
> manifest: `python -m driverguard.evaluation.uta_rldd manifest --root <thư mục UTA> --out data/uta_manifest.csv`.

## Lưu ý giấy phép

Feature trích xuất (số) vẫn là **dữ liệu dẫn xuất** từ dataset — giữ dataset Kaggle ở chế độ **Private**,
không public lại video hay feature của DMD (CC BY-NC-ND) hoặc State Farm (Competition Rules).
