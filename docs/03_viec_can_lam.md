# 03 — Việc cần làm (checklist cho nhóm)

> Lộ trình đang theo dõi từ 01/10/2026 nằm tại
> [14_model_roadmap.md](14_model_roadmap.md): tối ưu camera → ban đêm → EEG
> cảnh báo sớm → Agents → IoT. Checklist bên dưới là kế hoạch hệ thống ban đầu,
> giữ để tham khảo; trạng thái và thứ tự hiện tại theo roadmap mới.

Đánh dấu `[x]` khi xong. Các mục **[BẠN]** là việc máy của Claude không làm được (cần internet trên máy bạn,
tài khoản, chữ ký, hoặc con người thật).

## A. Cài đặt trên máy Windows (≈ 20 phút) — [BẠN]

```powershell
cd D:\code\DriverGuard
py -3.11 -m venv .venv            # MediaPipe hỗ trợ Python 3.9–3.12, KHÔNG dùng 3.13
.venv\Scripts\activate
pip install -e ".[api,eval,dev]"
python scripts\download_models.py # tải face_landmarker.task + efficientdet_lite0.tflite vào models\
pytest                            # kỳ vọng: tất cả pass (test_api cần fastapi — đã có trong [api])
```

- [ ] `driverguard simulate --scenario drowsy` → thấy CAUTION → WARNING → CRITICAL (MICROSLEEP) ở ~232 s
- [ ] `driverguard simulate --scenario narrow_eyes --set calibration.method=fixed` → báo động giả liên tục; bỏ `--set` → không còn (minh hoạ Hướng 1, **dữ liệu tổng hợp**)
- [ ] `driverguard run --source 0 --show` → webcam, thử nhắm mắt 2 s, cúi xuống 3 s, quay đầu. **Kiểm tra dấu yaw/pitch** (xem TODO trong `perception/head_pose.py`)
- [ ] Dashboard: terminal 1 `driverguard serve --source synthetic:demo`; terminal 2 `cd dashboard && npm install && npm run dev` → mở http://localhost:5173 (đăng nhập `manager/manager` hoặc `driver01/driver01`)
- [ ] (Tuỳ chọn) `docker compose up --build` → http://localhost:8000

Nếu có lỗi: gửi nguyên văn traceback cho Claude.

## B. Tải dữ liệu — [BẠN]

| # | Việc | Cách làm | Ghi chú |
|---|---|---|---|
| B1 | **UTA-RLDD** | Không cần tải về máy nếu chạy trên Kaggle (mục C). Nếu muốn bản chính thức: link Google Drive trên sites.google.com/view/utarldd | Rất lớn (~30 giờ video) — ưu tiên Kaggle |
| B2 | **DMD** | Vào dmd.vicomtech.org → "Get download links" → tải bộ **Drowsiness** và **Distraction** (RGB). Giải nén vào `data/raw/dmd/` | Chỉ 14 subject còn được phép dùng. **Gửi email hỏi license (mục D2)** |
| B3 | **YawDD** | Tạo tài khoản IEEE miễn phí → IEEE DataPort "YawDD: Yawning Detection Dataset" → tải → `data/raw/yawdd/` | Ưu tiên thấp (ngáp là chỉ số phụ) |
| B4 | **State Farm** | Kaggle → competition "state-farm-distracted-driver-detection" → tab Rules → **Accept**. Chưa cần tải về máy | Dùng trực tiếp trong notebook Kaggle ở GĐ1 |

## C. Chạy trên Kaggle — [BẠN] (hướng dẫn chi tiết: `kaggle/README.md`)

- [ ] C1. Tạo Kaggle dataset **Private** `driverguard-src` = zip thư mục `driverguard/`
- [ ] C2. Import `kaggle/01_extract_features.ipynb`, add input UTA-RLDD + `driverguard-src`, Internet ON
- [ ] C3. Kiểm tra ô manifest in ra **180 video, 60 subjects, folds [1..5]** — nếu không, báo lại cấu trúc thư mục cho Claude
- [ ] C4. Save & Run All với `FOLDS=[1]` → tải `features_uta_rldd_fold1.zip` → giải nén vào `data/features/uta_rldd/`
- [ ] C5. Lặp cho fold 2–5
- [ ] C6. Gửi cho Claude file `run_uta_rldd_fold1.json` (thời gian chạy, tỷ lệ phát hiện mặt) để kiểm tra chất lượng feature

## D. Giai đoạn 0 — việc không phải code (tuần 1–2)

- [ ] D1. Mang `docs/00_scope_mapping.md` cho GVHD duyệt (Quyết định #1)
- [ ] D2. Email info-dmd@vicomtech.org hỏi rõ giấy phép dữ liệu (MIT hay CC BY-NC-ND 4.0) → lưu phản hồi vào `docs/licenses/`
- [ ] D3. Hoàn thiện & in `templates/mau_don_dong_y.md`; chốt nơi lưu và thời hạn xoá
- [ ] D4. Phỏng vấn 5–8 người theo `templates/kich_ban_phong_van.md` (≥3 tài xế công nghệ, 2 đường dài, 2 chủ nhà xe)
- [ ] D5. Trả lời 7 câu hỏi cuối tài liệu GVHD (đồ án AI hay hệ thống? nhóm mấy người? ngân sách?…)
- [ ] D6. Phân công: perception/Android · engine/eval · backend/dashboard · dữ liệu & gán nhãn

## E. Bước code tiếp theo (Claude có thể làm khi bạn giao)

1. **GĐ1** — Sau A: chỉnh dấu yaw/pitch theo webcam thật; thêm hand landmarks cho điện thoại; notebook Kaggle đánh giá phone detector trên State Farm; bắt đầu project Android + unit test golden.
2. **GĐ1** — Parser annotation DMD (VCD) → `Interval` để đánh giá mức sự kiện trên xe thật.
3. **GĐ2** — Thí nghiệm Hướng 1 trên DMD + tập vàng: `adaptive` vs `fixed`, false alerts/giờ và recall theo người, có khoảng tin cậy.
4. **GĐ2** — `02_train_temporal.ipynb`: GRU/TCN trên feature, đối chứng với rule engine cùng split (Hướng 2); học trọng số hợp nhất (Hướng 4).
5. **GĐ3** — Ba con số generalization gap (Hướng 3); ma trận robustness; benchmark nhiệt 30 phút trên Android; test bắt gói tin.
6. **GĐ4** — Log vòng đời app (bật/tắt, che camera) cho metric Tầng 3: tỷ lệ vô hiệu hoá 7 ngày, tỷ lệ chấp nhận cảnh báo.
