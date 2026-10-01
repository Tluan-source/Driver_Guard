# Kế hoạch model camera, EEG và lái xe ban đêm

Ngày cập nhật: 01/10/2026. Phạm vi theo yêu cầu mới: hoàn thiện model camera
nhận diện dấu hiệu buồn ngủ, phản hồi nhanh và hoạt động trong thiếu sáng; sau đó
dùng EEG đồng bộ trong lab để huấn luyện khả năng cảnh báo sớm. Agents hỗ trợ và
IoT triển khai sau phần mô hình.

## Hiện trạng và mục tiêu

Repo có MediaPipe nhận diện khuôn mặt, đặc trưng mắt/miệng/tư thế đầu, temporal
trackers và luật cảnh báo; xem [kiến trúc hiện tại](01_architecture.md). Đã train
checkpoint camera v1: GRU nhân quả trên chuỗi đặc trưng, 354 clip của 59 người,
split 39/10/10 theo người. Test clip AUROC 0,6444 và balanced accuracy 0,55;
đây là prototype có thể test, chất lượng chưa đủ cho cảnh báo thực tế. Xem
[hướng dẫn sử dụng](11_camera_model_usage.md) và
[báo cáo camera v1](12_camera_model_results.md). Các checkpoint
[EEG v3.1](09_model_v3_results.md) là nghiên cứu EEG độc lập, chưa phải teacher đã
được chứng minh có thể dạy camera cảnh báo sớm. Bộ EEG hiện tại không có video
đồng bộ hoặc nhãn onset; không thể ghép các bản ghi không cùng người/thời điểm
để thực hiện luồng này.

## Tham chiếu video do người dùng cung cấp

Đã kiểm tra khung hình của file cục bộ
`FDown.vn_Tai_video_Facebook_720p (HD)_46fe.mp4` (7 phút 6,8 giây).
Đây là tham chiếu thiết kế, không phải corpus camera–EEG để train. Nhận xét dưới
đây dựa trên nội dung chữ/hình ở các mốc, không phải bản chép toàn bộ lời thoại:

| Mốc video | Nội dung thấy được | Áp dụng cho đề tài |
|---|---|---|
| 02:00 | “Buồn ngủ thì mở lại chậm”; hình mô phỏng theo Caffier/Schleicher | Thử động học mở lại mắt cùng thời lượng nhắm; không mặc định EAR bằng phần trăm che đồng tử |
| 04:00 | Điện cực O1, Oz, O2 | Các kênh ứng viên cho nghiên cứu EEG; chọn bằng validation và kiểm soát nhiễu |
| 04:15 | Alpha khác khi nhắm/mở mắt, nguồn EEGMMIDB | Nhắm mắt khi còn tỉnh cũng đổi EEG; alpha đơn lẻ không xác định onset buồn ngủ |
| 04:30–04:45 | Sleep-EDF, marker thiếp đi; mắt ghi là minh họa | Có thể nghiên cứu sinh lý giấc ngủ; không ghép minh họa mắt với EEG thành cặp dữ liệu thật |
| 05:10 | Teacher/student; ghi “sơ đồ đề xuất”, đường chậm mô phỏng trên phiên S45 | Mục tiêu EEG giám sát student camera-only; cần train và chứng minh trên cặp ghi đồng bộ thật |
| 05:25–05:30 | Cui 2022, Oz, nhiễu cơ 30–45 Hz | Không bỏ bước kiểm soát nhiễu EEG; không dùng gamma nhiễu cơ làm đáp án |
| 06:00–06:10 | Đường nhanh bắt nhắm mắt, đường chậm đọc buồn ngủ; mắt/cảnh báo mô phỏng, thời gian đánh lái Cao 2019 | Kết hợp nhánh phản ứng nhanh với nhánh temporal đã học; số phút minh họa không phải lead time của dự án |

Video kết hợp nhiều nguồn EEG, hành vi và mô phỏng. Không có cơ sở để coi các
nguồn này là cùng người/cùng phiên có camera đồng bộ. `learning/blink.py` hiện
có bộ đo mở lại mắt nhân quả nhưng chưa tích hợp vào checkpoint camera; đặc
trưng chỉ có sau khi chu kỳ hoàn tất, không đưa ngược về thời điểm trước đó.

Hướng nghiên cứu mới sẽ là **mô hình hai thang thời gian có giám sát EEG và
kiểm soát chất lượng ánh sáng**: nhánh nhanh bắt sự kiện mắt nguy hiểm, nhánh
chậm dự đoán nguy cơ onset từ quá khứ camera. So sánh camera thường, camera
thêm động học mở mắt, rồi camera có EEG supervision, ở cùng false alerts/giờ.
Đây là đóng góp dự kiến cần thí nghiệm, chưa khẳng định mới so với toàn bộ tài
liệu khoa học. Prototype 10 FPS chưa đủ để kết luận về pha mở mí mắt nhanh;
cần dữ liệu ít nhất 30 FPS và EEG đồng bộ cho thí nghiệm này.

Đích bàn giao là model camera có checkpoint, pipeline suy luận, điểm buồn ngủ,
trạng thái chất lượng cảm biến, cảnh báo và báo cáo metrics trên người chưa thấy.
Không có mặt hoặc ảnh quá tối phải trả trạng thái không đánh giá được, không suy
ra tài xế tỉnh táo.

## Giai đoạn 1: Model camera dùng được

1. Dùng Face Landmarker đã huấn luyện sẵn để nhận diện khuôn mặt, mắt, mũi,
   miệng và tư thế đầu. Kiểm tra confidence và chất lượng ảnh trước khi sử dụng
   đặc trưng. Tầng nhận diện này chưa tự dự đoán trạng thái buồn ngủ.
2. Huấn luyện một temporal head có checkpoint trên chuỗi đặc trưng mắt, chớp
   mắt, động học mở lại mắt, ngáp và tư thế đầu. Mỗi dự đoán chỉ dùng frame hiện
   tại/quá khứ; cửa sổ không vượt ranh giới phiên ghi. Chọn độ dài ngữ cảnh và
   model trên validation theo người.
3. Xuất điểm buồn ngủ và trạng thái chất lượng; nối điểm model với chính sách
   cảnh báo có hysteresis, dwell và cooldown. Đo riêng hành vi của model và
   toàn bộ pipeline cảnh báo để thấy tác động của thời gian chờ.
4. Bàn giao checkpoint camera, định nghĩa đặc trưng/tiền xử lý, split và hash dữ
   liệu, predictions ngoài tập train, metrics và lệnh chạy webcam/video.

Mục tiêu kỹ thuật ban đầu là **ít nhất 15 FPS và P95 độ trễ xử lý không quá
100 ms** trên laptop mục tiêu. Kiểm tra headless trên 100 frame clip RGB
224×224 ở 10 FPS cho P50 37,17 ms, P95 47,33 ms của perception + model + luật.
Chưa đo capture, hàng đợi, preview, âm thanh hoặc webcam ở độ phân giải thực;
chưa kết luận đạt mục tiêu toàn hệ thống. Đo độ trễ từ frame capture đến kết quả, thêm độ trễ đến phát
cảnh báo, FPS và frame bị bỏ; không để hàng đợi frame cũ tăng vô hạn. Tốc độ
suy luận khác với độ dài ngữ cảnh và thời gian tích lũy bằng chứng cảnh báo.

## Thiếu sáng và ban đêm là yêu cầu chính

Hai điều kiện cần phân biệt:

| Điều kiện | Phương án | Cần kiểm chứng |
|---|---|---|
| RGB thiếu sáng nhưng còn thông tin mặt | Tiền xử lý gamma/CLAHE có giới hạn, rồi trích đặc trưng và suy luận | Mất mặt, mắt/miệng, noise, motion blur và độ trễ khi bật/tắt tiền xử lý |
| Gần như không có ánh sáng khả kiến | Camera nhạy NIR và nguồn chiếu IR phù hợp ở giai đoạn phần cứng | Dữ liệu NIR thật, nhận diện mặt/mắt, kính gây phản xạ IR, gaze và khác biệt miền RGB/NIR |

Tiền xử lý RGB thiếu sáng được bổ sung ở phiên này chỉ là bước hỗ trợ ảnh;
camera v1 còn được train với exposure 0,6 ở tập train, nhưng hai bước này chưa
là bằng chứng đạt chất lượng ban đêm. Đánh giá độ sáng/chất lượng trên **frame gốc** và truyền trạng thái đó qua
pipeline: ảnh đã tăng sáng không được làm hệ thống tin rằng tín hiệu gốc đủ tốt.
Gamma/CLAHE không khôi phục chi tiết mắt đã mất trong tối hoặc blur.

Luồng hiện đã dùng `LowLightEnhancer` trước Face Landmarker. Các giá trị trong
`low_light` là mức pixel 0–255, không phải lux: mặc định chỉ tăng sáng khi mean
luma ảnh gốc từ 8 đến dưới 60 và tương phản percentile 95–5 ít nhất 8 mức. Ảnh
đen hoặc phẳng được giữ nguyên. Các ngưỡng này là seed kỹ thuật cần kiểm chứng
trên validation, không phải xác nhận ảnh đạt chất lượng. Trigger dùng toàn frame
có thể bỏ sót mặt tối khi nền/cửa sổ sáng; chưa có cơ chế exposure theo ROI mặt.
Phone detector vẫn nhận ảnh gốc, chưa được tối ưu hoặc kiểm chứng ban đêm.

Chạy thử luồng camera hiện có:

```powershell
.venv\Scripts\python.exe -m driverguard run --source 0 --show
.venv\Scripts\python.exe -m driverguard run --source 0 --show --set low_light.enabled=false
```

Hai lệnh dùng cùng luồng nhận diện/luật hiện tại để so sánh tiền xử lý; chưa phải
đường chạy checkpoint camera v1. Để test model đã học dùng lệnh trong
[hướng dẫn camera](11_camera_model_usage.md). Không giảm ngưỡng chất lượng ảnh gốc
để làm số dự đoán hợp lệ trông cao hơn.

Kiểm tra chức năng tái lập đã có tại `scripts/benchmark_low_light_camera.py`:
MediaPipe thật trên một ảnh giảm sáng nhân tạo, độc lập tracking ở từng điều kiện.
Ảnh đen không có mặt và chuyển sensor-degraded; ảnh tăng sáng vẫn bị loại nếu
quality gốc không đủ. P95 perception quan sát được trên ảnh 410×480 là khoảng
4,34–27,35 ms; số này không bao gồm capture, hàng đợi, khởi tạo và cảnh báo,
không phải FPS/độ trễ toàn hệ thống hay accuracy ban đêm. Kết quả cục bộ lưu ở
`outputs/night_camera_smoke.json`.

```powershell
.venv\Scripts\python.exe scripts/benchmark_low_light_camera.py --out outputs/night_camera_smoke.json
```

Face Landmarker hiện chưa được kiểm chứng trên ảnh IR/NIR của camera dự kiến.
Không mặc định khả năng RGB sẽ chuyển nguyên vẹn sang NIR. Cần thử tập thật,
đánh giá ảnh xám và cân nhắc model/finetune phù hợp nếu landmark thất bại.
Camera NIR có đèn IR vẫn là hệ thống camera-only khi triển khai, không cần EEG.

## Dữ liệu và đánh giá camera

- Thu hoặc dùng video thật ban ngày, thiếu sáng và ban đêm có quyền sử dụng,
  nhãn trạng thái/sự kiện, người/phiên và thông tin ánh sáng. Lưu manifest để
  kiểm tra phân bố, mất mặt, kính, tư thế, blur và nguồn camera. Dataset chỉ có
  ảnh rời không đủ để đánh giá dự đoán theo thời gian và cảnh báo.
- Chia theo người trước khi trích cửa sổ hoặc augmentation; toàn bộ phiên và
  biến thể của cùng người ở cùng tập. Các cửa sổ gần nhau không được xuất hiện
  ở cả train và test. Validation chọn model/ngưỡng; test đóng băng.
- Augmentation gamma, exposure, sensor noise và motion blur chỉ áp dụng vào
  train. Cùng một mẫu và biến thể của nó không đi qua các tập khác nhau.
  Ảnh làm tối nhân tạo không thay thế kiểm chứng trên video ban đêm thật.
- Báo cáo confusion matrix, macro-F1, AUROC/AUPRC, sensitivity, specificity và
  coverage theo người và theo nhóm ánh sáng. Giữ các cửa sổ không đánh giá được
  trong báo cáo; ghi rõ metrics trên mẫu hợp lệ và coverage toàn bộ.
- Đo false alerts/giờ, tỷ lệ phát hiện sự kiện, độ trễ cảnh báo và benchmark
  latency/FPS/frame drops riêng theo ánh sáng. So sánh bật/tắt tiền xử lý trên
  cùng clip để xác định hiệu quả và chi phí, không chỉ xem ảnh tăng sáng.

## Giai đoạn 2: EEG dạy camera cảnh báo sớm

Khởi tạo từ checkpoint camera đã train ở giai đoạn 1 và giữ student đầu vào
camera. Thu camera và EEG **cùng người, cùng phiên**, có timestamp đồng bộ và
kiểm tra sai số đồng bộ. Dữ liệu cần bao phủ cả thiếu sáng để student cuối không
chỉ được giám sát trong điều kiện lab sáng.

EEG đã kiểm soát chất lượng cung cấp tham chiếu sinh lý, kết hợp annotation
theo thời gian để định nghĩa onset suy giảm tỉnh táo. EEG có nhiễu và tham chiếu
onset cũng có bất định; không coi dự đoán của checkpoint EEG hiện tại là đáp án
chính xác sẵn có. Chuyên gia/protocol phải xác nhận nhãn trước khi dùng giám sát.

Thử fine-tune camera bằng nhãn onset hoặc teacher EEG+camera truyền tri thức
cho student camera-only. Teacher cần được kiểm chứng trên người chưa thấy;
teacher score giám sát mẫu train phải được tạo với tách người phù hợp. So sánh
camera trước fine-tune, camera dùng nhãn EEG và teacher/student để xác định EEG
có thêm lợi ích hay chỉ phản ánh dấu hiệu mắt đã có.

Mục tiêu nghiên cứu bổ sung là nguy cơ onset trong **10 hoặc 30 giây tới**.
Camera chỉ nhìn hiện tại/quá khứ; EEG tương lai có thể tạo nhãn của mục tiêu
forecasting khi train, nhưng không trở thành đầu vào student. Cửa sổ tương lai
không đủ độ dài cuối phiên phải được đánh dấu không có nhãn. Horizon là cấu
hình nghiên cứu, chưa phải cam kết cảnh báo trước được 10/30 giây.

Đo early lead time so với onset EEG, sensitivity tại các khoảng báo trước,
false alerts/giờ và phân bố lead time theo người/ánh sáng. Đo cả trường hợp
phát hiện sau onset; không chỉ báo số giây của các ca cảnh báo sớm thành công.
Các metrics này khác FPS/P95 latency của giai đoạn 1.

## Mô hình cuối và điều kiện bàn giao

Model cuối nhận camera, trả điểm buồn ngủ hiện tại và nguy cơ onset tương lai
nếu nhiệm vụ này được dữ liệu chứng minh; EEG chỉ phục vụ train/đánh giá trong
lab. Cần một cohort độc lập theo người, video ngày/đêm thật, cùng báo cáo chất
lượng cảm biến và độ trễ trên thiết bị mục tiêu trước khi kết luận đạt yêu cầu.

Ưu tiên triển khai: dữ liệu camera theo thời gian và checkpoint camera trước;
kiểm chứng thiếu sáng trong cùng giai đoạn; thu dữ liệu EEG-camera đồng bộ để
fine-tune sau đó. Thử NIR với phần cứng thật nằm trong bước IoT nhưng yêu cầu
dữ liệu/đánh giá NIR được xác định từ bây giờ. Agents không thay quyết định của
model hoặc chính sách cảnh báo.
