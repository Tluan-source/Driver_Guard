# Hướng nghiên cứu và phạm vi mô hình

Ngày thực hiện: 01/10/2026. Mục tiêu phiên này là một mô hình đã train, có thể dự đoán
và xuất dữ liệu để viết metrics. Agents là phần hỗ trợ ở phiên sau; IoT thực hiện tiếp theo.

## Quyết định dựa trên dự án hiện tại

Ưu tiên mô hình, dữ liệu và thực nghiệm. DriverGuard đã có MediaPipe, hiệu chỉnh EAR,
temporal trackers và luật cảnh báo; đây là nền tảng camera và vận hành sẵn có. Phần cần
bổ sung để làm đề tài là checkpoint học từ dữ liệu thật, quy trình đánh giá theo người
và bằng chứng so sánh với baseline. Agents phù hợp với giải thích cảnh báo và tổng hợp
chuyến đi khi đã có kết quả mô hình đo được.

Trong phiên này, mô hình EEG hoạt động độc lập với engine camera hiện tại. Không gọi
hai thành phần riêng biệt là mô hình đa phương thức đã được train hoặc kiểm chứng.

## Dữ liệu và nhiệm vụ được triển khai

Dùng [The original EEG data for driver fatigue detection](https://figshare.com/articles/dataset/5202739),
phiên bản 1, DOI `10.6084/m9.figshare.5202739.v1`, giấy phép CC BY 4.0.
Bộ dữ liệu có [bản mirror trên Kaggle](https://www.kaggle.com/datasets/jcxuitsme/eeg-driver-fatigue-detection);
script tải từ Figshare để lưu nguồn gốc, giấy phép, MD5 công bố và SHA-256 của archive.
SEED-VIG cần đăng ký/thỏa thuận sử dụng và chưa có dữ liệu được cấp trong phiên này,
nên không dùng làm dữ liệu train cho checkpoint bàn giao.

Theo [Min, Wang và Hu (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188756),
có 12 nam giới khỏe mạnh, 19-24 tuổi, lái xe trong bộ mô phỏng. Mỗi người có hai bản ghi
khoảng năm phút: `Normal state.cnt` và `Fatigue state.cnt`. Nhãn là trạng thái thí nghiệm
của **cả bản ghi**: bình thường = 0, mệt mỏi = 1; mọi cửa sổ trong bản ghi kế thừa nhãn đó.
Không có nhãn PERCLOS, thời điểm bắt đầu buồn ngủ hoặc video camera đồng bộ trong archive.

Đầu vào là 150 đặc trưng: 30 kênh EEG hiệu dụng × 5 dải tần delta/theta/alpha/beta/gamma.
Trích công suất phổ bằng Welch trên cửa sổ bốn giây không chồng lấn, tham chiếu trung bình
các kênh EEG, lấy `log10` công suất tính theo microvolt bình phương. Thứ tự là kênh trước,
dải tần sau; loại các kênh EOG, mastoid và kênh phẳng không sử dụng. Bộ đọc CNT dùng
`int32` vì header số mẫu của bản phát hành không phù hợp; độ dài giải mã phải khớp
khoảng năm phút ở 1000 Hz theo bài báo.

Đầu ra là **điểm trạng thái mệt mỏi trong [0, 1]**, kèm quyết định tại ngưỡng đã chọn
trên validation. Điểm này chưa được hiệu chỉnh như một xác suất và không phải chẩn đoán.

## Huấn luyện và đánh giá

1. Chia 12 người thành train/validation/test = **8/2/2**, seed 42. Hai trạng thái của cùng
   một người luôn ở cùng tập; không chia ngẫu nhiên các cửa sổ vào train và test.
2. Với nhiệm vụ fatigue, Ridge/TCN mặc định dùng `relative_log_power`: chia công suất
   mỗi dải cho tổng năm dải của cùng kênh, rồi lấy `log10`. Phép biến đổi theo từng cửa sổ
   không dùng nhãn hoặc người khác. Ước lượng mean/std từ các cửa sổ train sau biến đổi
   rồi lưu trong checkpoint; validation/test chỉ sử dụng các tham số này.
3. So sánh **tám họ ứng viên**: điểm trung bình nhãn train, Ridge trên công suất tương đối,
   Logistic Regression trên công suất tuyệt đối và tương đối, RBF-SVM trên công suất
   tuyệt đối và tương đối, TCN một cửa sổ tương đối và TCN ngữ cảnh tương đối.
   Ridge chọn alpha trong `{0.1, 1, 10, 100, 1000}` theo RMSE validation. Logistic chọn
   `C` trong `{0.1, 1, 10}`; SVM chọn cùng tập `C` và gamma trong `{scale, 0.01}`, theo
   AUROC validation rồi RMSE. Mỗi cách biểu diễn có scaler riêng chỉ fit trên train.
   TCN ngữ cảnh dùng tối đa tám cửa sổ quá khứ/hiện tại, tương đương khoảng 32 giây.
   Cửa sổ ngữ cảnh không đi qua ranh giới bản ghi và không dùng tương lai. Đầu bản ghi
   được đệm bằng mẫu đầu đã quan sát, chưa có đủ 32 giây lịch sử thật.
4. TCN dùng binary cross entropy; epoch tốt nhất được chọn theo validation RMSE và có
   early stopping. Chọn ứng viên cuối theo **AUROC validation cao nhất, rồi RMSE thấp nhất**.
   Chọn ngưỡng theo balanced accuracy validation; đóng băng trước khi evaluate test.
5. Xuất AUROC, AUPRC, balanced accuracy, macro-F1, precision, recall, confusion matrix,
   MAE/RMSE của score so với nhãn 0/1, metrics theo người và bootstrap theo người. MAE/RMSE
   ở đây không đo sai số PERCLOS hay mức mệt mỏi liên tục được đo trực tiếp.
6. Bản bàn giao cuối nằm ở **`models/eeg_vigilance_v2/model.pt`**. Lưu checkpoint từng ứng viên,
   split, manifest/hash, lịch sử train,
   predictions CSV và metrics JSON. Hướng dẫn tại [05_model_usage.md](05_model_usage.md),
   kết quả thực tế tại [06_model_results.md](06_model_results.md).

Lần chạy `v1` với công suất tuyệt đối chọn mô hình hằng số, được giữ lại làm đối chiếu
quá trình phát triển, không dùng làm bản dự báo cuối. Trước `v2`, đã thăm dò 119 cấu hình
trên train/validation; chuẩn hóa công suất tương đối và các baseline phân loại được đưa
vào pipeline từ bước này. Vì validation chỉ có hai người và đã được dùng để phát triển
nhiều cấu hình, kết quả validation có nguy cơ khớp riêng tập đó. Cần cross-validation
nhiều fold và dữ liệu độc lập để kiểm chứng trước khi kết luận khả năng tổng quát hóa.

Đây là đánh giá trên một phép chia theo người, có hai người test. Số cửa sổ lớn
không làm tăng số người độc lập; khoảng tin cậy bootstrap còn hạn chế với hai người.
Chưa có cross-validation nhiều fold, kiểm chứng trên đường thật hoặc phần cứng EEG trực tiếp.

## Hướng tính mới cần kiểm chứng

Giả thuyết nghiên cứu: **động học mở lại mắt có thể bổ sung cho đặc trưng EEG để phát hiện
suy giảm tỉnh táo, và mô hình có thể giảm phụ thuộc cảm biến bằng truyền tri thức**.

- Đã có utility đo thời gian mở lại, độ dốc mở lại và tỷ lệ thời gian mở/nhắm từ EAR chuẩn hóa.
  Chỉ nhận chu kỳ quan sát đầy đủ; mất dữ liệu hoặc đứt đoạn thời gian hủy chu kỳ. Đây là
  proxy hình học, không phải đo phần trăm che đồng tử. Chưa đưa vào model EEG hay đánh giá
  hiệu quả phân biệt trạng thái mệt mỏi; ảnh minh họa là nguồn cảm hứng cho giả thuyết.
- Cần thu EEG và RGB của cùng người, cùng phiên, timestamp đồng bộ trước khi fine-tune
  nhánh fusion. Không ghép camera của người này với EEG của người khác để tạo mẫu.
- Khi có dữ liệu đó: so sánh camera-only, EEG-only, fusion; thử quality gating,
  modality dropout, và teacher EEG+RGB truyền tri thức cho student camera-only.
- Báo cáo ablation và đánh giá trên người chưa thấy; đo độ trễ phát hiện và false alerts/giờ
  khi có nhãn sự kiện theo thời gian. Kiểm tra artifact mắt/chuyển động trong EEG và lỗi
  mất mặt, thiếu cảm biến. Chưa tuyên bố những hướng này là đóng góp đã được chứng minh.

Code và báo cáo được lưu Git. Dữ liệu gốc, đặc trưng và checkpoint nằm trong thư mục
ignored theo quy ước hiện tại; bản clone mới cần tải/train lại hoặc nhận checkpoint riêng.
