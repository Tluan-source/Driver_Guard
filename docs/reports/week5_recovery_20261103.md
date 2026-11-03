# Đối chiếu và khôi phục tuần 5 — 03/11/2026

Phần source và thực nghiệm CV chính của tuần 5 đã được khôi phục, kiểm tra và chạy
lại inference. **Chưa đủ bằng chứng để đánh dấu hoàn thành toàn bộ tuần 5**:
các bản sao hiện có thiếu 44 model đối chứng cũ, low-light và dữ liệu DROZY.
Mục tiêu accuracy >85% vẫn chưa đạt theo chính kết quả Kaggle.

Nguồn yêu cầu: [sheet tiến độ, tab tuần 5](https://docs.google.com/spreadsheets/d/1xTmUC8TSeNlwzOtknd8n4HYEc-2OISQRW6JcvNxYzI8/edit?gid=911502963#gid=911502963),
docs trong Git và source/artifact của
[notebook Kaggle tuần 5](https://www.kaggle.com/code/thanhluanchimto/driverguard-week-5-subject-cv/output).

## Đã khôi phục và kiểm tra

| Hạng mục trong báo cáo | Bằng chứng trên máy | Trạng thái |
|---|---|---|
| Source train và protocol tuần 5 | 139 file source/config/notebook; đủ 138 file provenance train khớp hash bản gốc | Đã khôi phục |
| 5 fold, 50 người development | Mỗi fold 32 train /8 inner validation /10 outer; không trùng người giữa split | Đã kiểm tra |
| Khóa 10 người UTA test gốc | `02,07,08,09,15,16,18,41,49,57` không có trong fold/feature bundle | Đã kiểm tra |
| 30 run GRU/DANN ×3 seed ×5 fold | Checkpoint hữu hạn; hash khớp metadata; đủ CSV/history/protocol | Đã kiểm tra |
| Metrics và paired CI | Tái tính từ CSV development; khớp điểm số và CI Kaggle | Đã kiểm tra |
| Feature gốc | 151 NPZ của 50 người, byte/hash giữ nguyên | Đã khôi phục |
| Replay CPU 30 checkpoint | 960 cửa sổ inner validation lấy mẫu; sai lệch tối đa `4.529953e-6`, tolerance `1e-5` | Đạt kiểm tra nhất quán |
| Replay qua API Model Lab | Clip `UTA_S11_0`, 204/204 thời điểm khớp CSV; sai lệch tối đa khoảng `1e-6` | Đạt kiểm tra nhất quán |
| Runtime 120 giây | Checkpoint thật + tín hiệu mô phỏng: warmup 1200 bin, mất mặt ngừng score, reset xóa lịch sử | Đã kiểm tra |
| Dashboard/Model Lab | Dashboard đọc metadata; 30 fold không tham gia lựa chọn cuối; console replay/live local | Đã khôi phục UI |
| 44 checkpoint đối chứng cũ | Không có trong hai ZIP hoặc checkout | Chưa khôi phục |
| Low-light 20 video ×7 điều kiện | Có script/config; thiếu video nguồn và 140 NPZ cũ | Chưa tái kiểm chứng |
| DROZY 36 phiên /14 người | Có downloader/audit; thiếu dữ liệu để kiểm lại intake | Chưa tái kiểm chứng |

Không dùng lại con số 365 test trong sheet như kết quả kiểm thử mới: bộ test đó
không có trong source bundle. Bộ test Git hiện có cùng regression test khôi phục
được chạy riêng; kết quả kiểm thử ghi ở phần xác minh bên dưới.

## Số liệu đã tái tính

Task **alert/drowsy**, cửa sổ 120 giây, 10 Hz, stride 25 bin, quyết định **argmax**;
không tune threshold. Ensemble là trung bình xác suất đều của seed 42/43/44,
ghép đúng clip và timestamp; coverage báo riêng.

| Mô hình | Accuracy | CI95% theo người | Macro-F1 | Coverage |
|---|---:|---|---:|---:|
| GRU | 76,0246% | [70,0361%; 81,5931%] | 75,9661% | 98,2007% |
| GRU+DANN | 75,4240% | [69,4512%; 81,1563%] | 75,3757% | 98,2007% |

Mỗi ensemble nhận 19.812/20.175 cửa sổ. DANN − GRU = **−0,6006 điểm phần trăm**,
paired CI95% **[−1,3889; +0,1065]** trên 50 người/19.812 thời điểm chung. CI chứa 0;
theo quy tắc đã khai báo, tiếp tục hướng GRU. Đây là lựa chọn hướng kiến trúc,
không phải chọn một checkpoint CV làm model triển khai cuối.

Dữ liệu này từng dùng phát triển kiến trúc. Các file `test_windows.csv` ở mỗi run
chỉ chứa outer fold của 50 người development. Test gốc và cohort độc lập chưa mở.
Accuracy một clip hoặc mô phỏng không thay thế số liệu tổng hợp trên đây.

## Hai bản sao gốc

| ZIP | SHA-256 | Nội dung |
|---|---|---|
| `driverguard-week5-results-20261019.zip` | `e570c21c72d7d99612437d51440d7d6d163085299e41b6ae3c3e8fe0a975daf6` | 369 file: 30 run, fold manifests, summary/comparison |
| `driverguard-week5-dev-20261020.zip` | `34a488865fd9f2fafed097b23e1be46a44b6619125e7376550d5d78ca3df1138` | 291 file: 139 source và 151 NPZ +manifest |

ZIP đã qua CRC. Bản gốc giải nén ở `models/recovered_week5/` và
`data/recovery_week5/`, kèm manifest hash. Source nguyên trạng ở
`data/recovery_week5/driverguard-source-week5-20261019/`; overlay source vào working
tree có backup ở `outputs/backup_before_original_source_overlay/`. Các sửa chữa
runtime/UI sau khôi phục không thay thế bản source gốc dùng train.

Model Lab dùng bản checkpoint sao chép ở
`models/from_kaggle/week5_gru_dann/fold*_*/inference/model.pt`. Feature hoạt động ở
`data/features/week5_development/`: chỉ đổi đường dẫn trong manifest cho máy local,
không thay byte NPZ hoặc fold manifest gốc. Source overlay và bước relocation được
ghi trong `outputs/recovery_week5/`.

## Xác minh có thể chạy lại

Kết quả mới trên máy này: **307 test passed, 0 failed, 0 skipped** (247,41 giây),
dashboard TypeScript/Vite build thành công, Ruff trên các file sửa chữa/khôi phục
được liệt kê trong `verification.json` đạt, compile Python và `git diff --check` đạt.
Có một cảnh báo deprecation của Starlette TestClient/httpx; không có lỗi test.
Đã kiểm trực tiếp UI bằng trình duyệt: đăng nhập demo manager, bảng CV 30 run,
replay, mô phỏng, reset và dừng. Webcam thật chưa được mở để đo hiệu năng.

Tại root project, dùng Python của `.venv`:

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_week5_recovery
.\.venv\Scripts\python.exe -m scripts.replay_week5_recovery
.\.venv\Scripts\python.exe -m scripts.check_week5_runtime
.\.venv\Scripts\python.exe -m scripts.evaluation.summarize_subject_cv --root models/recovered_week5 --protocol configs/training/evaluation_protocol_v1.json --out outputs/recovery_week5/cv_comparison_recomputed.json
.\.venv\Scripts\python.exe -m pytest -q -o addopts=
```

File bằng chứng mới:

- `outputs/recovery_week5/artifact_audit.json`: 369 hash, 30 checkpoint, CSV và split.
- `outputs/recovery_week5/cpu_replay.json`: 151 hash NPZ và 960 cửa sổ của 30 model.
- `outputs/recovery_week5/cv_comparison_recomputed.json`: điểm số/CI tái tính.
- `outputs/recovery_week5/runtime_check.json`: replay qua API, warmup, mất mặt, reset.
- `outputs/recovery_week5/verification.json`: kết quả kiểm thử/build/UI mới nhất.

Để nhập lại ZIP vào thư mục mới (không ghi đè bản gốc):

```powershell
.\.venv\Scripts\python.exe -m scripts.recover_week5 --help
.\.venv\Scripts\python.exe -m scripts.import_week5_inputs --help
```

Hai importer xác minh CRC, kiểm tra đường dẫn và hash trước khi dùng. Dữ liệu/model
và ZIP được Git ignore. Code chưa được push lên GitHub trong phiên khôi phục này.

## Phần còn thiếu và bước tiếp theo

1. Tìm bản sao 44 model cũ kèm protocol/prediction/feature tương ứng để so sánh lại
   cùng timestamp và tách task binary/3 lớp. Nếu train lại, phải ghi đó là run mới.
2. Khôi phục video validation/140 NPZ low-light. Không thể tạo lại ảnh tối từ NPZ
   chứa feature đã trích; không coi RGB giảm sáng là dữ liệu đêm thật.
3. Khôi phục/tải đủ DROZY rồi audit file theo phiên/người. CamSync/interpIndices,
   teacher EEG và ablation nằm ở công việc tiếp theo; chưa có bằng chứng đã làm.
4. Chuyển sang [kế hoạch tuần 6–7](../ROADMAP.md) sau khi chốt được trạng thái các
   artifact còn thiếu. Không đánh dấu đạt >85% nếu chưa có kết quả đúng protocol.
