# Tuần 6: công cụ, bằng chứng và phần còn cần người thực hiện

Cập nhật 05/11/2026. Đối chiếu các dòng 34–38 của tab tuần 5 trong
[sheet báo cáo](https://docs.google.com/spreadsheets/d/1xTmUC8TSeNlwzOtknd8n4HYEc-2OISQRW6JcvNxYzI8/edit?gid=911502963).

## Trạng thái

| Hạng mục | Kết quả hiện có | Điều kiện chưa hoàn tất |
|---|---|---|
| Phân tích lỗi | 1.359 dòng thống kê feature; khóa hai giả thuyết cải tiến GRU | Chưa có kết quả CV mới |
| A2 | 40 video nguồn khớp hash; 40 clip 10 giây, sửa rotation, xuất H.264; AI xem 4.000 frame, lập 45 interval | AI là một lượt đề xuất, chưa có hai lượt người review và phân xử |
| DROZY | Giải mã đủ 36 video/14 người; kiểm EDF/TXT, xuất mapping có mask cho từng frame/row | Chỉ 29 phiên đạt ngưỡng phân tích đã cấu hình; không chứng nhận onset/nhãn EEG |
| Chuẩn bị ablation | 30 job = 2 biến thể × 5 fold × 3 seed; hai smoke CPU chạy được | Full CV chỉ chạy trên Kaggle T4 ×2; chưa chạy đợt mới |
| Đánh giá thực tế | Xác minh thiết bị, chuẩn bị protocol và công cụ khóa cohort theo người | Chưa tuyển người/thu phiên ngày–đêm hoặc xác nhận nhãn độc lập |

**Chưa nghiệm thu toàn bộ tuần 6.** Thiếu nhãn/người thật không được thay bằng
dữ liệu giả hoặc kết quả smoke. Accuracy development cũ vẫn là GRU 76,02%,
DANN 75,42%; mục tiêu accuracy >85% chưa được xác nhận.

## Chuẩn bị môi trường

Chạy từ thư mục repository:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api,eval,train,dev]"
```

PyAV 17.1 được khai báo trong nhóm `eval` để giải mã PTS, đọc display rotation
và xuất clip H.264 phát được trên trình duyệt. Dữ liệu, video, checkpoint,
CSV gán nhãn, link riêng của người review và log đều giữ trong thư mục bị Git bỏ qua.

## A2: xem và lưu review

Video nguồn development đã trích chọn nằm dưới `data/raw/uta_rldd_full`.
Chưa giải nén hoặc mở payload của mười người test gốc.

```powershell
.\.venv\Scripts\python.exe -m scripts.data.materialize_week6_review
.\.venv\Scripts\python.exe -m scripts.evaluation.review_week6
```

Server chỉ nghe trên `127.0.0.1:8011`. Mở
`outputs/week6/a2_review/local_review_links.json` để lấy link riêng cho A, B
và người phân xử. Link được tạo lại mỗi lần khởi động server. A/B không nhìn
thấy biểu mẫu của nhau; người phân xử chỉ nhận hai bản đối chiếu sau khi cả
hai hoàn tất. Sau khi tắt máy, cần chạy lại lệnh server để tiếp tục review.

Luân và Huy xem độc lập theo [protocol A2](16_camera_annotation_protocol.md),
ghi mã người, mốc thời gian, bằng chứng và lý do uncertain. Không xem bản AI
trước nếu muốn giữ lượt review độc lập. Các interval phải phủ [0,10), không
bỏ trống/chồng lấn; có thể thêm interval khi visibility/trạng thái đổi.
Khi thiếu context, giữ uncertain và yêu cầu đoạn nguồn dài hơn.

Biểu mẫu tự lưu vào `reviewer_a.csv`, `reviewer_b.csv`, `adjudication.csv`.
Mỗi lần lưu có bản cũ theo hash trong `history/` và nhật ký thay đổi. Server
không tự tạo nhãn từ EAR, score model hoặc nhãn nguồn.

### Lượt AI đã thực hiện

Theo yêu cầu người dùng, AI đã xem contact sheet chứa **đủ 100 frame/clip ở
10 Hz**, có crop cố định để nhìn rõ mặt và ảnh toàn khung đối chiếu. R015,
R037, R039, R040 được xem thêm ảnh lớn. Đây không phải xem video liên tục,
không có đủ context dài, và trợ lý đã biết một phần ngữ cảnh triển khai nên
không tuyên bố review hoàn toàn mù.

Kết quả cấp clip: 28 `alert` (chưa thấy dấu hiệu buồn ngủ rõ), 12 `uncertain`,
0 `drowsy_visible`. Sau tách interval: 297,8 giây alert và 102,2 giây uncertain.
Không suy ra sinh lý, không kết luận nhãn video nguồn sai, không dùng nhóm
clip ưu tiên lỗi để ước lượng tỷ lệ nhãn sai toàn bộ corpus.

File cục bộ: `outputs/week6/a2_review/ai_review/ai_review.csv` và `summary.json`.
`reviewer=AI_assistant_v1`, `review_kind=AI_provisional`,
`ready_for_training_labels=false`; không đổi biểu mẫu người thật hay nhãn train.
Bản CSV đề xuất AI được lưu cùng source tại
[week6_ai_review_20261105.csv](results/week6_ai_review_20261105.csv), kèm
[bằng chứng tổng hợp](results/week6_summary_20261105.json). Không có ảnh/video
hoặc link riêng của người review trong hai file này.

## DROZY: căn chỉnh có mask và loại phiên không đủ độ phủ

`DROZY.zip` chính thức nằm ở `data/DROZY.zip`; bản giải nén ở
`data/raw/drozy/DROZY`. Bản 2018 thực tế dùng EDF và TXT, không phải các MAT
được giả định trong template trước khi tải dữ liệu. README xác định
`interpIndices` tham chiếu frame video **đánh số từ 1**, cột thứ hai bằng 0
khi không dùng frame thứ hai; PSG ghi ở 512 Hz.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.audit_drozy_release
.\.venv\Scripts\python.exe -m scripts.evaluation.align_drozy
```

Intake giải mã toàn bộ video, so số frame/timestamp, kiểm PTS và hash nguồn.
Mapping sử dụng crossing qua 0 V của Cam-Sync, chấp nhận ghép duy nhất trong
15 ms; frame thiếu hoặc tranh cùng một xung bị mask, không làm lệch ghép các
frame phía sau. Timestamp thực nghiệm khác PTS phát lại của video mã hóa.

Policy lưu tại `configs/training/drozy_alignment_v1.json`: độ phủ frame ≥98%,
độ phủ interpolation row ≥95%, khoảng nội suy ≤150 ms. Do gốc grid chưa được
xác nhận là row/30 hay (row+1)/30, mỗi row phải phù hợp cả hai trong 100 ms.
Mapping lưu thời gian PSG thực từ Cam-Sync, không tự chọn gốc có lỗi đẹp hơn.

29 phiên đạt policy cho **phân tích mapping có mask**. Loại bảy phiên:
`2-3`, `4-2`, `5-2`, `5-3`, `6-3`, `7-2`, `8-2`. Đây là loại theo tiêu chí
độ phủ/gap hiện tại, không khẳng định toàn bộ dữ liệu các phiên đó hỏng.
Riêng `5-2`, row cuối tái dựng ở 599,715 s, lệch 285 ms so với mốc 600 s.
`6-3` và `8-2` có nhiều xung không khớp.

Đầu ra `outputs/week6/drozy_maps/*.npz` chứa index PSG zero-based, mask và
residual. Vị trí bị loại là `-1`/`NaN`; không điền EEG giả. Báo cáo
`summary.json` gắn hash intake/config/map. **Không có nhãn onset sinh lý,
không bật teacher–student** từ riêng bước này.

## Ablation camera đã chuẩn bị

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.run_week6_ablation --help
```

Kế hoạch giữ task binary alert/drowsy, context 120 giây, argmax, 50 người
development; fold theo người trùng mốc tuần 5, seed 42/43/44. Hai biến thể
thay từng yếu tố: input noise 0,05 và dropout 0,3. Chọn epoch bằng inner
validation; outer development dùng so sánh, không phải test độc lập.

Notebook `kaggle/05_week6_ablation.ipynb` kiểm hash source/151 NPZ, chuẩn bị
30 job trên T4 ×2. Hai smoke CPU chỉ kiểm pipeline và checkpoint, không dùng
điểm smoke làm kết quả nghiên cứu. Không có đợt Kaggle mới được gửi tự động.

Đã sửa điều kiện đạt mục tiêu trong trainer: chỉ accuracy >0,85; macro-F1
cao không thay thế accuracy. Dashboard/Model Lab nhận provenance CV mới và
tiếp tục loại checkpoint từng fold khỏi ứng viên nghiệm thu cuối.

## Cohort ngày/đêm độc lập

Protocol đề xuất 20 người mới, tách 12 development/8 test theo người, hai
phiên RGB ngày/đêm liên tục ít nhất 600 giây/người. Thiết kế này chưa được
thay bằng cohort diễn xuất hoặc DROZY NIR trong lab.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.audit_week6_cohort
```

Điền `outputs/week6/cohort/participants.csv` bằng ID giả danh P001…P999,
mã đồng thuận và xác nhận không thuộc UTA, rồi chạy `--freeze` **trước** khi
thu/xem video. Split và hash roster/protocol được khóa. Điền `sessions.csv`
theo phiên thực, đường dẫn dưới `data/raw/week6_cohort`; không bịa người,
đồng thuận, phiên hoặc nhãn. Công cụ chỉ kiểm metadata/attestation, không
mở video/nhãn test và không chứng minh độc lập chỉ từ một ô xác nhận.

## Tổng hợp trạng thái

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.week6_status
.\.venv\Scripts\python.exe -m pytest
```

`outputs/week6/status.json` kiểm lại hash bằng chứng, báo riêng AI/human,
mapping và tình trạng cohort. Chỉ nâng trạng thái khi dữ liệu/người review
thực sự đủ; không đánh dấu tuần 6 hoàn tất vì đã viết xong công cụ.
