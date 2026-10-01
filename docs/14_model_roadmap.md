# Lộ trình model DriverGuard và bảng theo dõi

Cập nhật: 01/10/2026. Đây là bảng theo dõi công việc hiện tại, theo thứ tự
đã thống nhất: **camera đạt chất lượng → fine-tune ban đêm → EEG giám sát
cảnh báo sớm → đóng gói model → Agents → IoT**.

**Việc tiếp theo: A1 — audit dữ liệu camera và lỗi trên tập development.**
Phiên này hoàn tất kế hoạch; các mục triển khai bên dưới chưa được chạy.
Chuẩn bị dữ liệu ban đêm/EEG được làm sớm để phát hiện thiếu dữ liệu trước
khi tới giai đoạn huấn luyện tương ứng.

## 1. Bảng tiến độ chính

| Mốc | Đầu ra phải có | Điều kiện hoàn thành | Trạng thái |
|---|---|---|---|
| P0. Chốt hiện trạng và lộ trình | Baseline v1, metrics, roadmap này | Có artifact và tiêu chí đo rõ ràng | **XONG** |
| A. Tối ưu camera ban ngày | Checkpoint `camera_day_v2`, báo cáo theo người, benchmark webcam | Accuracy và macro-F1 ≥85%; hướng tới 90%; đạt kiểm tra tốc độ/chất lượng bên dưới | **TIẾP THEO: A1** |
| B. Fine-tune thiếu sáng/ban đêm | Checkpoint `camera_day_night_v3`, báo cáo riêng từng miền ánh sáng | Đạt mục tiêu ở các miền được hỗ trợ, giữ chất lượng ban ngày | Chờ A; chuẩn bị dữ liệu từ sớm |
| C. EEG giám sát cảnh báo sớm | Teacher/nhãn EEG đã kiểm chứng và student camera-only | Chứng minh lợi ích cảnh báo sớm so với camera v3 ở cùng mức báo sai | Chờ B và dữ liệu đồng bộ |
| D. Đóng gói model cuối | Checkpoint, inference, model card, metrics tái tạo được | Chạy độc lập camera-only, giới hạn sử dụng và bằng chứng được ghi đầy đủ | Chờ C |
| E. Agents hỗ trợ | Đặc tả Agents dựa trên metadata model | Triển khai sau khi chốt model | Sau D |
| F. IoT/phần cứng | Tích hợp thiết bị, camera/NIR, benchmark thiết bị thật | Kiểm chứng trên cấu hình phần cứng đích | Sau E |

Các tên checkpoint mới trong tài liệu là **đầu ra dự kiến**, chưa tồn tại.
Thử NIR sớm nếu có thiết bị/dữ liệu; bước F là tích hợp sản phẩm phần cứng.

## 2. Điểm xuất phát đã xác minh

Camera v1 dùng MediaPipe + GRU 32 hidden units, 19 đặc trưng, 50 mẫu lịch sử
ở 10 Hz. Dữ liệu gồm 354 clip, 59 người; split train/validation/test 39/10/10.
Train có augmentation exposure 0,6. Nhãn theo video, chưa có nhãn onset.

| Metric test v1 | Theo clip | Theo cửa sổ nhân quả |
|---|---:|---:|
| Accuracy | 55,00% | 58,06% |
| Macro-F1 | 53,96% | 57,05% |
| AUROC | 0,6444 | 0,6392 |
| Recall buồn ngủ | 70,00% | 73,33% |
| Specificity | 40,00% | 42,78% |

Coverage v1 là 100% các cửa sổ test đã đủ lịch sử, chưa tính warmup. P95
perception + model + luật trên clip 224×224 là 47,33 ms; chưa có benchmark
capture-to-result trên webcam thật. Chi tiết ở [kết quả v1](12_camera_model_results.md).

Giữ v1 và predictions để đối chiếu. Test v1 đã được xem; tiếp tục dùng làm
mốc lịch sử, không chọn hyperparameter/ngưỡng theo kết quả này. Cần cohort
test mới, chưa dùng trong phát triển, để xác nhận phiên bản cuối. Thêm clip
của người đã có trong train không tạo ra người test độc lập.

## 3. Tiêu chí đo thống nhất

- **Mục tiêu chính A/B:** accuracy ≥0,85 và **macro-F1 ≥0,85** trên dự đoán
  theo cửa sổ chỉ dùng hiện tại/quá khứ, người test chưa thấy; 0,90 là mốc
  nâng cao. Đây là mục tiêu cần thực nghiệm chứng minh, chưa phải kết quả.
- Chốt nhãn, độ dài/bước cửa sổ, trọng số theo người, split và định nghĩa
  mẫu hợp lệ trước thí nghiệm. Báo cả pooled metrics và subject-macro;
  clip metrics là chỉ số phụ để đối chiếu v1, không thay cho cảnh báo online.
- Báo accuracy, balanced accuracy, macro-F1, recall/specificity từng lớp,
  AUROC/AUPRC, confusion matrix, CI bootstrap theo người và coverage.
  Đánh giá thêm trên video liên tục với tỷ lệ tỉnh/buồn ngủ thực tế hơn.
- Coverage phải có mẫu số gồm cả các thời điểm dự kiến dự đoán; tách warmup,
  mất mặt, thiếu sáng và các lần từ chối. Mức khởi điểm đề xuất là ≥95% sau
  warmup trong điều kiện camera tuyên bố hỗ trợ; kiểm tra lại ở pilot và
  khóa trước test cuối. Không giảm cổng chất lượng để đạt số coverage.
- **Tốc độ:** mục tiêu ≥15 FPS xử lý camera và P95 capture-to-result ≤100 ms
  trên laptop đích, ghi cấu hình/độ phân giải, frame drops và tuổi frame.
  Đo riêng tới phát cảnh báo, thời gian khởi động và history warmup.
  Nhánh model có thể cập nhật 10 Hz; đó là tần số khác với FPS toàn pipeline.
- Đánh giá cảnh báo gồm event recall và false alerts/giờ trên video liên
  tục. Chốt mức báo sai chấp nhận được bằng pilot/validation trước test;
  chưa lấy false positive theo clip làm false alerts/giờ.
- Epoch, ngưỡng, calibration, augmentation và chính sách cảnh báo được
  chọn bằng train/validation. Nếu xem test rồi sửa model, lần xác nhận tiếp
  cần dữ liệu test mới; không gọi đó là cùng một test chưa thấy.

## 4. Mốc A — tối ưu model camera

- [ ] **A1. Audit dữ liệu và lỗi development.** Kiểm tra manifest/hash,
  nguồn video, người/phiên, FPS, clip trùng/gần trùng, crop mặt, mask và
  phân bố nhãn. Chọn khoảng 40 clip train/validation cân bằng hai lớp để
  xem lỗi tiêu biểu; ghi nhãn nghi ngờ, blink/ngáp, tư thế, kính và ánh sáng.
  Dùng false positive/false negative trên development để phân tích; không
  mở thêm test để chọn hướng sửa. Đầu ra: báo cáo audit và danh sách ưu tiên.
- [ ] **A2. Cải thiện dữ liệu và protocol.** Chốt định nghĩa “buồn ngủ” thay
  vì mặc định mọi frame của video đều mang cùng trạng thái. Đánh dấu vùng
  không chắc chắn; lập mẫu annotation cần người xác nhận. Ưu tiên video
  liên tục dài hơn và nguồn camera/người đa dạng, giữ nhóm người/phiên khi
  split. Tìm cohort test mới; kiểm tra quyền sử dụng trước đưa vào corpus.
  Đầu ra: manifest version mới, thống kê chất lượng, split cố định.
- [ ] **A3. Chạy thí nghiệm có đối chứng.** Bắt đầu bằng đặc trưng temporal
  và baseline nhỏ, rồi so GRU/TCN theo bảng bên dưới. So lỗi theo người và
  tính ổn định qua seed, không chỉ lấy lần chạy tốt nhất. Đầu ra: experiment
  log, predictions validation và checkpoint ứng viên.
- [ ] **A4. Chọn model và ngưỡng.** Chọn bằng validation theo người; khóa
  checkpoint, threshold và preprocessing. Đo điểm xác suất/calibration nếu
  muốn diễn giải score như xác suất. Đầu ra: model card và cấu hình inference.
- [ ] **A5. Xác nhận chất lượng/tốc độ.** Đánh giá trên cohort test mới khi
  đủ dữ liệu, benchmark webcam/video liên tục và các tình huống mất tín hiệu.
  Nếu chưa đạt mục tiêu, ghi rõ thiếu hụt rồi quay về A1/A2/A3. Không đánh
  dấu A hoàn thành chỉ vì đạt 85% ở train, validation hoặc clip chọn lọc.

### Thứ tự thí nghiệm A3

| ID | Thí nghiệm | Câu hỏi cần trả lời |
|---|---|---|
| CAM-00 | V1 đã đóng băng | Mốc chất lượng, tốc độ và lỗi hiện tại là gì? |
| CAM-01 | Thống kê temporal + logistic/boosting nhỏ | Feature mắt/miệng/đầu có đủ thông tin trước khi tăng độ phức tạp model? |
| CAM-02 | GRU + temporal features/masks, regularization | Lịch sử và feature cải thiện khả năng tổng quát theo người không? |
| CAM-03 | TCN nhân quả, cùng dữ liệu/features với GRU | Kiến trúc temporal nào tốt hơn khi giữ protocol giống nhau? |
| CAM-04 | Encoder nhỏ pretrained cho ROI mắt/miệng + temporal head | Chỉ mở nhánh này nếu audit cho thấy geometry thiếu thông tin và dữ liệu/compute đủ |

Feature ứng viên: thống kê EAR/MAR, tỷ lệ/thời lượng nhắm mắt, blink rate,
PERCLOS-proxy, ngáp, head motion và mask chất lượng. Động học mở mí nhanh
cần nguồn ≥30 FPS thật; không nội suy bộ 10 FPS rồi coi là đã đo được.
So context 5/10/20 giây khi có clip đủ dài; không pad/lặp clip 10 giây để
tạo bằng chứng lịch sử 20 giây. Mọi chuẩn hóa theo người chỉ dùng dữ liệu
quá khứ được phép, không dùng nhãn hoặc toàn phiên test.

Đợt đầu giới hạn khoảng 12 cấu hình trên CAM-01/02/03, sau đó lặp hai ứng
viên tốt với ba seed cố định. Nếu cải thiện không nhất quán trên validation,
ưu tiên audit/nhãn/dữ liệu thay vì mở rộng tìm kiếm hyperparameter. Chốt
ngân sách GPU/thời gian khi audit tài nguyên; mỗi run lưu thời gian và bộ nhớ.

## 5. Mốc B — fine-tune ban đêm

- [ ] **B1. Chuẩn bị dữ liệu từ sớm.** Thu thập/kiểm tra nguồn RGB thiếu sáng
  thật và NIR, gắn modality/ánh sáng/người/phiên. DROZY là ứng viên NIR trong
  lab; UL-DD cần xin truy cập và không có EEG. Kiểm tra cả landmark, kính,
  blur và phản xạ; ánh sáng nhân tạo giảm exposure chỉ là augmentation.
- [ ] **B2. Kiểm tra mốc trước fine-tune.** Đo checkpoint A trên validation
  ngày/thiếu sáng/NIR; so bật/tắt enhancement trên cùng clip và cùng nhãn.
  Nếu landmark không hoạt động trên NIR, sửa tầng nhận diện trước temporal head.
- [ ] **B3. Fine-tune kết hợp ngày–đêm.** Khởi tạo checkpoint A, giữ dữ liệu
  ban ngày khi train, cân bằng người/điều kiện; checkpoint mới có input
  contract và feature manifest riêng. Chọn ngưỡng bằng validation, không
  dựa vào nhãn ánh sáng suy ra nhãn buồn ngủ.
- [ ] **B4. Xác nhận từng điều kiện.** Mục tiêu ≥85% accuracy/macro-F1 cho
  từng miền được tuyên bố hỗ trợ; báo coverage và tốc độ riêng. So với A
  trên ngày để phát hiện quên kiến thức; chỉ kết luận hỗ trợ NIR/tối sâu khi
  có dữ liệu thật phù hợp. Dữ liệu NIR lab không nghiệm thu được lái xe đêm thực tế.

Đầu ra B: checkpoint ngày/đêm, so sánh trước–sau fine-tune, ablation enhancement,
metrics theo ánh sáng/người và danh sách điều kiện chưa hỗ trợ.

## 6. Mốc C — EEG dạy camera cảnh báo sớm

- [ ] **C1. Kiểm tra dữ liệu đồng bộ từ sớm.** Kiểm tra DROZY: EDF/video,
  timestamp, missing frames, FPS, license, kênh EEG, nhãn KSS/PVT. Giữ dữ
  liệu gốc; không ghép EEG của Cao hoặc Sleep-EDF với UTA không cùng phiên.
- [ ] **C2. Chốt định nghĩa onset và annotation.** Phân biệt suy giảm tỉnh
  táo với microsleep; xác định event, vùng bất định và tiêu chí EEG có
  kiểm soát nhiễu. KSS theo phiên hoặc PERCLOS không tự thành onset EEG.
  Cần protocol chuyên môn/GVHD và người xác nhận nhãn trước khi công bố lead time.
- [ ] **C3. Train/kiểm chứng teacher EEG.** So baseline bandpower với CNN
  gọn theo các paper Cui/ID3RSNet; kiểm tra nhiễu cơ/mắt và domain/kênh đo.
  Teacher/target tạo cho student-train phải tránh rò rỉ người/phiên và nhãn
  validation/test; dùng out-of-fold hoặc teacher huấn luyện độc lập phù hợp.
- [ ] **C4. Fine-tune student camera-only từ B.** Đối chứng: camera B,
  camera fine-tune cùng dữ liệu cặp với nhãn thông thường, camera có nhãn
  EEG, và teacher distillation nếu teacher đủ chất lượng. Giữ task hiện
  trạng, thêm head onset-risk nếu dữ liệu hỗ trợ. Xét horizon 10/30 giây như
  giả thuyết; dữ liệu cuối phiên thiếu horizon phải được đánh dấu không có nhãn.
- [ ] **C5. Kiểm chứng cảnh báo sớm.** Chỉ camera hiện tại/quá khứ là input
  của student. Ở cùng false alerts/giờ, báo event sensitivity, tỷ lệ bắt được
  trước onset, lead time, phát hiện muộn/bỏ sót, coverage và CI theo người.
  Báo riêng ngày/đêm. Không chỉ lấy trung bình lead time ở các ca thành công.

Điều kiện qua C: teacher/nhãn đủ tin cậy, có event test độc lập và camera
student cho lợi ích có bằng chứng so với đối chứng đã fine-tune cùng dữ liệu.
Không mặc định sẽ báo trước được 10/30 giây. Nếu dữ liệu hiện có chưa đủ
annotation/onset, đầu ra chỉ là proof of concept; ghi rõ C chưa hoàn thành.
Các paper và mức truy cập: [13_eeg_related_papers.md](13_eeg_related_papers.md).

## 7. Mốc D — bàn giao model cuối

- [ ] Checkpoint camera-only + checksum + input contract + environment/config.
- [ ] Lệnh webcam/video, JSONL/CSV và ví dụ tính metrics; test mất mặt/tối/gap.
- [ ] Predictions test, metrics theo người/ánh sáng, CI, runtime và model card.
- [ ] Ablation chứng minh tác dụng dữ liệu, temporal features, night fine-tune,
  EEG supervision; công bố kết quả âm nếu một thành phần không giúp.
- [ ] Ranh giới sử dụng: camera/độ phân giải/ánh sáng hỗ trợ, warmup, tình
  huống từ chối đánh giá và điều kiện cần kiểm chứng thêm.

Code/config/report lưu Git; dữ liệu và trọng số theo chính sách lưu trữ,
giấy phép và checksum riêng. Agents sẽ đọc metadata của model sau mốc này.

## 8. Cách cập nhật để theo dõi dễ

Mỗi phiên chọn một task đang mở, đánh dấu bắt đầu/kết thúc, gắn link artifact
và ghi quyết định. Chỉ tick hoàn tất khi có đầu ra kiểm tra được. Mỗi run có
`run_id`, Git SHA, dataset/split hash, feature contract, seed, cấu hình,
checkpoint hash, validation metrics, runtime và lý do giữ/bỏ. Chỉ điền test
metrics khi thực sự chạy đợt xác nhận đã đóng băng.

| Ngày | Task/run | Trạng thái | Bằng chứng / quyết định |
|---|---|---|---|
| 01/10/2026 | CAM-00 | Xong baseline | [Kết quả v1](12_camera_model_results.md), chưa đạt mục tiêu |
| 01/10/2026 | P0 | Xong kế hoạch | Roadmap này; triển khai tiếp theo bắt đầu từ A1 |
| — | A1 | Tiếp theo | Audit train/validation, xác định lỗi và chất lượng nhãn |
| — | B1/C1 | Chuẩn bị sau A1 | Xác minh nguồn dữ liệu và điều kiện sử dụng trước tải/thu lớn |

### Gói việc của phiên triển khai tiếp theo

1. Chạy audit manifest/feature, tài nguyên CPU/GPU/dung lượng; giữ bản v1.
2. Phân tích lỗi development và lập danh sách clip cần xác nhận nhãn.
3. Chốt protocol A2, nguồn mở rộng dữ liệu và kế hoạch cohort test mới.
4. Khi audit không phát hiện blocker, chạy CAM-01 để có đối chứng đầu tiên.
5. Cập nhật bảng trên với metric thực, artifact và task kế tiếp.

Tôi phụ trách audit tự động, pipeline, thí nghiệm, metrics và tài liệu.
Bạn/GVHD tham gia xác nhận định nghĩa nhãn/onset và bố trí dữ liệu/người hoặc
thiết bị khi cần; các thủ tục truy cập bên ngoài sẽ được chuẩn bị cụ thể.
Chưa cần mua EEG/IoT để bắt đầu A1. Hướng dẫn test hiện có:
[11_camera_model_usage.md](11_camera_model_usage.md).
