# Kết quả mô hình EEG v2

Ngày thực hiện: 01/10/2026. Đã có pipeline tải dữ liệu, trích EEG, train, checkpoint,
predict từ CNT/NPZ và xuất metrics. Mô hình có thể dùng để làm thực nghiệm; độ tin cậy
hiện tại chưa đáp ứng cảnh báo tài xế trong thực tế.

## Dữ liệu Và Giao Thức

Nguồn: [Driver Fatigue EEG trên Figshare](https://doi.org/10.6084/m9.figshare.5202739.v1),
giấy phép CC BY 4.0. [Bài báo gốc](https://doi.org/10.1371/journal.pone.0188756).
Có [mirror Kaggle](https://www.kaggle.com/datasets/jcxuitsme/eeg-driver-fatigue-detection).
Đã tải SEED-VIG qua Kaggle và viết adapter, nhưng không train nguồn đó do điều khoản gốc
yêu cầu thỏa thuận cấp phép học thuật.

12 nam giới, 19–24 tuổi, thí nghiệm mô phỏng. Mỗi người có khoảng 5 phút normal và 5 phút
fatigue. Nhãn là trạng thái thí nghiệm ở mức bản ghi, không phải nhãn mắt/PERCLOS/onset.
24 bản ghi → 1.800 cửa sổ không chồng lấn, mỗi cửa sổ 4 giây; 900 normal, 900 fatigue.

- Train: subject 03, 04, 06, 07, 08, 09, 10, 12; 1.200 cửa sổ.
- Validation: subject 01, 05; 300 cửa sổ.
- Test: subject 02, 11; 300 cửa sổ.
- Seed 42; cả hai trạng thái của một người nằm trong cùng tập. Không nối ngữ cảnh giữa các bản ghi.

CNT đọc bằng int32, tính lại số mẫu và kiểm tra giới hạn kích thước file. 30 kênh EEG,
loại EOG/mastoid/kênh không sử dụng; 1.000 Hz; common average reference; Welch 1 giây,
50% overlap; log10 công suất delta/theta/alpha/beta/gamma. Trích từng cửa sổ, không dùng
mẫu tương lai. Lưu channel order, thông số, SHA256 nguồn và feature.

## Checkpoint Đã Chọn

`models/eeg_vigilance_v2/model.pt`: **Logistic Regression C=10 với relative log power**.
Input vẫn là 150 giá trị log10 công suất tuyệt đối từ extractor. Predictor tự chuyển
sang log công suất tương đối từng kênh và chuẩn hóa bằng thống kê train. Không chuẩn hóa
lại đầu vào trước khi gọi predictor.

- Kích thước checkpoint: 21.601 byte.
- SHA256: `cc238d60b8a8c096455e8d43e044f64f30bc45bb5ec5ed8bbddefb17bc7b4b7a`.
- Score trong [0,1], chưa được hiệu chỉnh xác suất.
- Ngưỡng chọn trên validation: `0.000292514750414`, giữ nguyên khi evaluate test.
- Python 3.10.11, NumPy 2.2.6, Torch 2.14.1+cpu, scikit-learn 1.7.2; MNE 1.12.1.

Giữ baseline v1 với công suất tuyệt đối, chọn mean baseline trên validation. Việc phát triển
v2 dựa trên train/validation: đã thăm dò 119 cấu hình biểu diễn/classifier, không mở NPZ
test trong bước thăm dò. Script tái lập: `scripts/explore_eeg_validation.py`. Chỉ có hai
người validation nên việc thử nhiều cấu hình có nguy cơ overfit validation.

Pipeline v2 fit chuẩn hóa trên train; chọn Ridge alpha, Logistic C, SVM C/gamma và ngưỡng
trên validation. TCN chọn epoch theo validation RMSE. Chọn ứng viên cuối theo AUROC
validation cao nhất, hòa thì RMSE thấp nhất; khóa trước khi evaluate test của v2.
V1 cũng đã được evaluate trong quá trình phát triển, vì vậy không trình bày toàn bộ dự án
như một thí nghiệm đăng ký trước chỉ chạm test đúng một lần.

## Metrics

Các dòng đối chứng giữ ngưỡng validation riêng. **Không đổi model sang dòng có test tốt hơn**.

| Ứng viên | Val AUROC | Test AUROC | Test Balanced Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| Mean baseline | 0,5000 | 0,5000 | 0,5000 | 0,3333 |
| Ridge, relative power | 0,8370 | 0,4935 | 0,5333 | 0,4608 |
| Logistic, absolute power | 0,2964 | 0,5667 | 0,5900 | 0,5072 |
| SVM, absolute power | 0,7669 | 0,6799 | 0,5967 | 0,5577 |
| **Logistic, relative power: đã chọn** | **0,9242** | **0,6464** | **0,5000** | **0,3448** |
| SVM, relative power | 0,8204 | 0,5373 | 0,5567 | 0,5199 |
| TCN một cửa sổ, relative power | 0,8342 | 0,5757 | 0,5967 | 0,5615 |
| TCN 8 cửa sổ, relative power | 0,7390 | 0,4363 | 0,4967 | 0,4965 |

Model đã chọn: accuracy 0,5000; precision 0,5000; recall 0,9867; specificity 0,0133;
AUPRC 0,5594; MAE 0,3794; RMSE 0,5289. MAE/RMSE là sai số score so với target 0/1,
không phải sai số dự đoán PERCLOS. Recall cao đi cùng 148/150 cửa sổ normal bị báo fatigue.

Confusion matrix, hàng = thật, cột = dự đoán:

| | Normal | Fatigue |
|---|---:|---:|
| Normal | 2 | 148 |
| Fatigue | 2 | 148 |

AUROC theo người test: subject 02 = 0,8274; subject 11 = 0,4923. Bootstrap theo người,
300 lần, khoảng percentile 95% cho pooled AUROC: [0,4923; 0,8274]. Chỉ hai người test
khiến khoảng này rất thô; balanced accuracy CI [0,5; 0,5] không chứng minh mô hình ổn định.

## Artifact Và Kiểm Tra

Trong `models/eeg_vigilance_v2/`: `model.pt`, checkpoint đối chứng, `metrics.json`,
`selection.json`, `split.json`, `dataset_manifest.json`, history TCN,
`predictions_test.csv`, CSV validation/test từng ứng viên, `evaluation.png`.
Model và dữ liệu được Git ignore; code, tests và báo cáo được lưu Git.

Đã kiểm tra dự đoán CNT gốc và NPZ cho subject 03 normal: cùng 75 cửa sổ, output khớp
hoàn toàn. Predictor không cần nhãn để suy luận. Evaluate từ chối dữ liệu test bị thay đổi
so với fingerprint checkpoint. Source hash của các module learning khớp phiên train v2.

Toàn bộ pytest: **146 passed**, một cảnh báo deprecation của FastAPI/Starlette có sẵn.
Ruff cho toàn bộ module learning, script và test mới: passed.

Tạo lại biểu đồ bằng `python scripts/plot_model_results.py --run models/eeg_vigilance_v2`.
Xem lệnh sử dụng và Python API trong [hướng dẫn](05_model_usage.md).

## Ý Nghĩa Cho Đề Tài

Kết quả ủng hộ tập trung vào **mô hình, biểu diễn, artifact và tổng quát hóa theo người**.
TCN chưa vượt baseline học máy trên tập này; không có bằng chứng rằng thêm độ sâu hay
Agents sẽ cải thiện khả năng phát hiện fatigue. EEG chỉ là một nhánh độc lập hiện tại.

Audit train/validation phát hiện nhiễu điện cực FT7 rất lớn ở subject 06; average reference
có thể lan nhiễu sang các kênh khác. Chưa lọc artifact có hệ thống. Chưa có RGB đồng bộ,
đánh giá ngoài đường thật, nhãn sự kiện hay hiệu chỉnh xác suất. Không báo false alerts/giờ
hoặc onset delay từ nhãn mức bản ghi.

Hướng tiếp theo có thể kiểm chứng: reference/artifact rejection bền vững, hiệu chỉnh
cá nhân với protocol calibration riêng, LOSO và một tập test độc lập mới. Đặc trưng thời
gian mở lại mắt đã có code/test nhưng chưa chứng minh lợi ích; fusion EEG-camera và
teacher–student cần dữ liệu cùng người và cùng thời điểm. Agents và IoT chưa triển khai.
