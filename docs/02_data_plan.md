# 02 — Kế hoạch dữ liệu, giấy phép và chia tập

> Thông tin truy cập/giấy phép kiểm tra ngày 28/09/2026. Giấy phép có thể đổi — ghi lại ngày tải và lưu
> bản chụp điều khoản vào `docs/licenses/` (tạo khi tải).

## 1. Nguồn dữ liệu và việc mỗi nguồn đo được

| Nguồn | Truy cập | Giấy phép / điều khoản | Nhãn thật sự có | Đo được | KHÔNG đo được |
|---|---|---|---|---|---|
| **UTA-RLDD** — 60 người, 180 video ~10 phút, ~30 giờ RGB | Google Drive từ trang chính thức (sites.google.com/view/utarldd). Có bản mirror cộng đồng trên Kaggle (`rishab260/uta-reallife-drowsiness-dataset`) | Trang chính thức chỉ yêu cầu trích dẫn bài báo (arXiv 1904.07312), **không nêu license rõ** → chỉ dùng nghiên cứu, không phân phối lại | **Mức video**: 0 tỉnh / 5 lơ mơ / 10 buồn ngủ. 5 fold chính thức × 12 người. 51 nam / 9 nữ | Phân loại mức video; xu hướng PERCLOS-proxy/blink theo nhãn | Eye-state F1, blink F1, onset delay, false alerts/giờ |
| **DMD** (Vicomtech) — bản 2026: chỉ RGB xe thật, 14 subject | dmd.vicomtech.org → "Get download links" (≥ 18 tuổi, học thuật) | Website ghi **CC BY-NC-ND 4.0**; README GitHub ghi MIT → **mâu thuẫn, phải email info-dmd@vicomtech.org** và lưu phản hồi làm phụ lục | Annotation theo thời gian (VCD ≥ 5.0): distraction, drowsiness, gaze, head pose, hands | Đánh giá mức **sự kiện** độc lập người dùng trên xe thật (LOSO, báo khoảng tin cậy) | Tổng quát hoá rộng (chỉ 14 người) |
| **YawDD** — ngáp / nói / bình thường, 2 vị trí camera, xe đỗ | IEEE DataPort, Open Access, cần tài khoản IEEE (miễn phí). DOI 10.21227/1g5m-rv87 | Open Access ≠ được dùng thương mại → kiểm tra điều khoản DataPort | Mức clip | Module ngáp (chỉ số phụ) | Mọi thứ khác |
| **State Farm Distracted Driver** — ảnh tĩnh, 10 lớp hành vi | Kaggle competition (phải bấm "I Understand and Accept" Rules) | **Competition Rules** — chỉ dùng theo điều khoản cuộc thi | Nhãn lớp ảnh (c1–c4: nhắn tin/gọi điện) + `driver_imgs_list.csv` (subject) | Độ chính xác phát hiện điện thoại mức ảnh, chia theo tài xế | Chiều thời gian |
| **Tập vàng tự quay** (bắt buộc) | Nhóm tự quay bằng đúng điện thoại mục tiêu | Theo đơn đồng ý của người tham gia | Nhãn **khoảng sự kiện** do nhóm gán | Onset delay, false alerts/giờ, ma trận robustness, cross-domain | — |

## 2. Chia tập — luôn theo NGƯỜI, không bao giờ theo frame

| Dataset | Giao thức | Code |
|---|---|---|
| UTA-RLDD | 5 fold chính thức, lần lượt 1 fold test / 4 fold train | `evaluation/uta_rldd.py` |
| DMD | Leave-one-subject-out (14 người) → báo **khoảng tin cậy** | `evaluation/splits.py` `leave_one_subject_out` |
| YawDD | Theo người | `group_kfold` |
| State Farm | Theo `subject` trong `driver_imgs_list.csv` | `group_kfold` |
| Tập vàng | 70/15/15 theo người **hoặc** chỉ dùng làm test cross-domain | `driver_split` |
| Cross-domain | Tune trên DMD (+UTA) → test zero-shot trên tập vàng | Hướng 3 |

Quy tắc chống "tune trên test": chọn điểm vận hành trên **validation** bằng `operating_curve.sweep` +
`pick_operating_point`, **đóng băng config** (commit hash), rồi chạy test **một lần**. Báo cáo đường cong
recall theo false alerts/giờ thay cho ngưỡng đạt/không đạt cứng (GVHD 8.3).

## 3. Tập vàng tự quay

- **Quy mô**: 8–12 người, tổng 90–120 phút; ưu tiên đa dạng giới tính, có người đeo kính cận, có người mắt một mí.
- **Thiết bị**: chính điện thoại mục tiêu, camera trước, gắn giá táp-lô như khi chạy bản đồ, 30 fps ≥ 720p.
- **Đồng ý**: ký `templates/mau_don_dong_y.md` **trước** khi quay. Không quay người không ký (kể cả người đi cùng xe).
- **An toàn**: các kịch bản nhắm mắt / điện thoại / cúi đầu **chỉ quay khi xe đứng yên** (như YawDD) hoặc trong mô phỏng. Không bao giờ yêu cầu người lái thật nhắm mắt khi xe chạy.
- **Lưu trữ**: ổ mã hoá trên máy của nhóm; **không** upload lên Kaggle/Drive công khai; trích feature bằng `driverguard extract` tại máy; xoá video gốc theo thời hạn trong đơn.
- **Kịch bản (ma trận robustness 19 mục)**: mặt chính diện sáng tốt · chớp mắt tự nhiên · nhắm lâu có chủ ý · nhắm lặp lại · ngáp · nói chuyện · liếc gương nhanh · cúi nhìn lâu · điện thoại trên tay · điện thoại trên giá · tay che mặt · kính cận · ngược sáng mạnh · cabin thiếu sáng · ra khỏi khung hình · che camera · rút camera · FPS 30→10 · mất mạng.

### Gán nhãn

- Gán **khoảng sự kiện** (start/end ms), không gán từng frame; định dạng CSV trong `evaluation/labels.py`
  (`video_id,start_ms,end_ms,label,annotator`). Công cụ gợi ý: CVAT hoặc Label Studio (chế độ video timeline), xuất ra CSV.
- Hai người gán độc lập trên **15%** video → báo **Cohen's kappa** (`labels.cohen_kappa`).
- Ngân sách: ~8–12× thời lượng video → **16–24 giờ công** cho 120 phút. Phân công người chịu trách nhiệm.

## 4. Việc chạy trên Kaggle vs trên máy

| Việc | Ở đâu | Vì sao |
|---|---|---|
| Trích feature UTA-RLDD (~30 giờ video) | **Kaggle** (`kaggle/01_extract_features.ipynb`) | Nặng CPU, dataset có sẵn trên Kaggle, không cần tải ~chục GB về máy |
| Đánh giá phone detector trên State Farm | **Kaggle** (GĐ1, notebook sẽ viết) | Dataset gắn thẳng vào notebook |
| Trích feature DMD / YawDD | Kaggle (upload **Private**) hoặc máy | Tuỳ băng thông; giữ Private vì giấy phép |
| Trích feature tập vàng | **Chỉ trên máy nhóm** | Dữ liệu khuôn mặt người tham gia — không đưa lên cloud |
| Replay, tune ngưỡng, hiệu chỉnh, GRU/TCN nhỏ | Máy (laptop) | Feature nhỏ, chạy vài giây–vài phút |
