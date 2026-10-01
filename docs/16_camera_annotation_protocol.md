# A2: protocol rà nhãn camera và mở rộng dữ liệu

Cập nhật 01/10/2026. Đây là protocol v1 cho audit development; chưa phải nhãn
onset EEG hoặc bộ test cuối. Gói đã tạo:
`outputs/camera_annotation_a2/`, 40 clip, tổng 400 giây, mọi interval còn
`unreviewed`. Video và biểu mẫu giữ cục bộ.

## 1. Rà soát 40 video

Người rà soát chỉ mở `videos/R001.mp4` đến `R040.mp4` và `annotations.csv`.
Thứ tự đã xáo trộn, tên video ẩn mã nguồn/nhãn, biểu mẫu không chứa nhãn gốc
hoặc score. `private_manifest.json` dành cho người quản lý dữ liệu để ghép
lại nguồn sau review; không dùng file này khi gán nhãn.

Xem trọn clip ở tốc độ gốc; xem lại đoạn cần kiểm tra. Một hàng là interval
`[start_s, end_s)`. Có thể thêm hàng khi trạng thái/dấu hiệu/visibility đổi.
Các interval phải phủ hết clip, không chồng lấn, không bỏ khoảng trống.
Thời gian tính từ đầu clip; độ phân giải nguồn là 0,1 giây.

| Cột | Giá trị và ý nghĩa |
|---|---|
| `review_id` | Mã R001–R040; giữ nguyên khi thêm interval |
| `start_s`, `end_s` | Thời điểm đầu/cuối interval, trong [0,10] của clip hiện tại |
| `state` | `alert`: chưa thấy dấu hiệu buồn ngủ rõ trong context; `drowsy_visible`: có tổ hợp dấu hiệu rõ theo protocol; `uncertain`: không đủ bằng chứng |
| `eye_closure` | `none`, `blink`, `sustained`, `uncertain`; ghi thời lượng quan sát trong notes |
| `yawn`, `head_nod` | `absent`, `present`, `uncertain` |
| `visibility` | `usable`, `occluded`, `unusable`, `uncertain` |
| `illumination` | `adequate`, `dim`, `uncertain`; không suy ra nhãn drowsiness từ độ sáng |
| `glasses` | `absent`, `present`, `uncertain` |
| `review_status` | Chỉ đổi sang `reviewed` sau khi xem interval/context |
| `reviewer` | Mã người gán nhãn, bắt buộc với hàng reviewed |
| `notes` | Bằng chứng nhìn thấy, lý do uncertain và timestamp; bắt buộc với hàng reviewed |

`alert` ở đây là nhãn **không có dấu hiệu rõ trên camera**, không chứng minh
người tỉnh về mặt sinh lý. `drowsy_visible` cần bằng chứng ngoài blink/ngáp
đơn lẻ: ví dụ mắt đóng/mở chậm lặp lại cùng gật đầu và suy giảm biểu hiện tỉnh
táo trong context. Trường hợp khó phân biệt với nói, chủ động nhắm mắt hoặc
nhìn xuống phải giữ `uncertain`; ghi riêng hành vi quan sát được.

Mắt mở trong clip nhãn nguồn buồn ngủ không tự thành nhãn sai. Không dùng EAR,
threshold hay score model để tạo đáp án cho chính model. Không gọi các dấu
hiệu camera này là onset sinh lý. Khi visibility không usable, state phải
uncertain. 10 FPS chỉ hỗ trợ proxy thời lượng; không đo động học mí mắt nhanh.

Ưu tiên hai người rà độc lập bằng hai bản CSV, không xem kết quả của nhau.
So agreement theo thời gian quan sát; người phụ trách/GVHD phân xử bất đồng
và chốt định nghĩa dấu hiệu. Một bản reviewed mới chỉ là một vòng review,
không tự động thành dữ liệu train đã nghiệm thu.

## 2. Công cụ đã triển khai

Kiểm tra biểu mẫu hiện tại hoặc từng bản review:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_camera_annotation.py validate --raw data/raw/camera_uta_pilot/manifest.json --package outputs/camera_annotation_a2/private_manifest.json --annotations outputs/camera_annotation_a2/annotations.csv
```

Công cụ kiểm tra danh mục nhãn, interval, độ phủ clip, bằng chứng/reviewer,
checksum video và provenance. Test clips bị cấm trong gói development.
`schema_valid=true` chỉ nói biểu mẫu hợp lệ về cấu trúc; nếu còn unreviewed,
status vẫn là `review_pending`. Khi reviewed hết, status là
`review_complete_requires_adjudication`. Công cụ không kiểm chứng chuyên môn
và không tự thay nhãn training.

Tái tạo gói trong thư mục mới:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_camera_annotation.py prepare --raw data/raw/camera_uta_pilot/manifest.json --queue outputs/camera_review_a1/review_queue.csv --out outputs/camera_annotation_a2_reproduce --seed 42
```

Sau adjudication, giữ nguyên manifest v1; xuất manifest phiên bản mới với
nhãn gốc, nhãn reviewed, uncertainty, phiên bản protocol và checksum CSV.
Không gán lại interval uncertain thành tỉnh/buồn ngủ. Report phải so cả protocol
v1 và nhãn mới phù hợp; không gọi tăng điểm do đổi task là cải thiện trên v1.
Gói được chọn ưu tiên lỗi nên không dùng để ước lượng tỷ lệ nhãn sai toàn corpus.

## 3. Nguồn dữ liệu kế tiếp và cohort độc lập

| Nguồn | Vai trò | Việc cần chốt trước đưa vào corpus |
|---|---|---|
| UTA-RLDD gốc/video dài hơn | Context 10/20 giây, kiểm tra timing nguồn | Quyền dùng và ID recording; người trùng corpus chỉ dùng development |
| DMD | Ứng viên camera xe thật và annotation temporal | Xác minh lại access/điều khoản mâu thuẫn đã ghi trong kế hoạch dữ liệu; chưa tải |
| YawDD | Kiểm chứng module ngáp | Điều khoản DataPort và split theo người; không coi ngáp là ground truth drowsiness |
| Cohort tự thu | Kiểm tra full-frame camera, đa dạng người/camera/ánh sáng | Protocol/GVHD, đồng thuận, định nghĩa nhãn; dành riêng người test chưa phát triển |
| DROZY | NIR và video–EEG đồng bộ ở B/C | KSS/PVT không tự thành onset; kiểm tra đồng bộ và quyền sử dụng |

Trạng thái access/điều khoản các nguồn ứng viên chưa được xác minh lại trong
phiên này. Tham chiếu [kế hoạch dữ liệu](02_data_plan.md) và
[paper EEG](13_eeg_related_papers.md) để kiểm tra nguồn chính thức.

Gợi ý pilot cohort mới: 12–20 người ngoài UTA, ít nhất hai phiên/người,
video liên tục vài phút, camera full-frame và các điều kiện có/không kính,
ánh sáng/góc đầu khác nhau. Đây là đề xuất thiết kế, chưa phải dữ liệu hiện có
hoặc bảo đảm đủ lực thống kê. Tách người development/test trước khi xem video;
giữ bản test riêng và không dùng để lựa chọn model/ngưỡng.

Dữ liệu diễn xuất chỉ kiểm tra phản ứng trước dấu hiệu; phải gắn cờ simulated
và không dùng để chứng minh onset sinh lý. Cảnh nhắm mắt/gật đầu chủ động chỉ
thu khi xe đứng yên hoặc trong mô phỏng. Nhãn fatigue thực và onset EEG cần
protocol phù hợp ở bước C.

Điều kiện hoàn thành A2: review được phân xử, manifest version mới có
interval/uncertainty và provenance, nguồn mở rộng hợp lệ, split cố định và
kế hoạch test độc lập có dữ liệu cụ thể. **Hiện mới có gói review và protocol;
A2 còn đang thực hiện.**
