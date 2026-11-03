# ruff: noqa: ISC004 -- long table cells are split across lines on purpose.
"""Wording of the week 1-3 progress reports (Vietnamese, as submitted to the supervisor)."""
from __future__ import annotations

SUPERVISOR = "Bùi Mạnh Quân"
MEMBERS = "Nguyễn Thành Luân & Nguyễn Đình Huy"

WEEK1 = {
    "subtitle": "Xác định đề tài, đối tượng sử dụng và hướng nghiên cứu",
    "period": "Tuần 1",
    "goals": [
        "Xác định vấn đề buồn ngủ, mất tập trung khi lái xe; nhu cầu của tài xế và người quản lý đội xe.",
        "Chốt phạm vi hệ thống: camera tại thiết bị, cảnh báo tại chỗ, metadata cho dashboard; "
        "EEG là hướng hỗ trợ huấn luyện sau.",
        "Khảo sát dữ liệu, công nghệ và protocol đánh giá theo người; lập lộ trình từ baseline tới model cải tiến.",
    ],
    "tasks": [
        ("Phạm vi dự án", "Tách perception, xử lý thời gian, model, cảnh báo, API/dashboard và Edge/IoT.",
         "Luân và Huy", "Đầu tuần 1", "Đã xác định", "Sơ đồ kiến trúc và phạm vi MVP.",
         "Edge/IoT triển khai sau khi model được kiểm chứng."),
        ("Bài toán và người dùng", "Mô tả tài xế và quản lý đội xe; lập giả thuyết về báo động giả, "
         "quyền riêng tư và theo dõi chuyến đi.", "Huy; Luân phối hợp", "Giữa tuần 1", "Đã nghiên cứu",
         "Persona và kịch bản phỏng vấn.", "Chưa có xác nhận từ khảo sát thực địa."),
        ("Giải pháp tham khảo", "Khảo sát DMS camera, tín hiệu mắt/miệng/đầu và cách cảnh báo.",
         "Luân và Huy", "Giữa tuần 1", "Đã tổng hợp", "Bảng so sánh giải pháp và khoảng trống nghiên cứu.",
         "Cần đối chiếu bằng thực nghiệm."),
        ("Hướng model", "Chọn baseline camera theo chuỗi GRU; định hướng thêm DANN để giảm lệ thuộc "
         "người/miền và EEG supervision về sau.", "Luân", "Cuối tuần 1", "Đã chọn hướng",
         "Lộ trình GRU → GRU+DANN → EEG đồng bộ.", "Đây là kế hoạch, chưa phải kết quả train."),
        ("Công nghệ", "Chọn Python, OpenCV/MediaPipe, PyTorch, FastAPI/WebSocket, React và Kaggle.",
         "Luân; Huy phối hợp", "Cuối tuần 1", "Đã xác định",
         "Kiến trúc xử lý camera tại thiết bị, chỉ gửi metadata.", "Chưa benchmark thiết bị đích."),
        ("Nguồn dữ liệu", "Khảo sát UTA-RLDD, NTHU, DROZY và EEG fatigue; ghi yêu cầu nhãn, phiên, timestamp.",
         "Huy; Luân phối hợp", "Cuối tuần 1", "Đã lập danh mục", "Ưu tiên dữ liệu camera UTA cho baseline.",
         "Kiểm tra quyền sử dụng trước khi đưa lên cloud."),
        ("Thiết kế đánh giá", "Chia theo người trước khi cắt cửa sổ; chọn model/ngưỡng trên validation; "
         "giữ test khóa.", "Luân và Huy", "Cuối tuần 1", "Đã có protocol",
         "Bộ chỉ số accuracy, macro-F1, AUROC, coverage; dự kiến false alerts/giờ.",
         "Nhãn clip không cho phép kết luận cảnh báo sớm."),
    ],
    "task_height": 68,
    "problems": [
        ("Phạm vi camera, EEG, Edge/IoT rộng.", "Ưu tiên baseline và đánh giá camera; chia EEG, Edge thành mốc sau.",
         "Xin duyệt phạm vi và đóng góp chính.", ""),
        ("Nhu cầu người dùng đang là giả thuyết.", "Phỏng vấn tài xế/quản lý theo kịch bản, lưu bằng chứng.",
         "Xin góp ý mẫu khảo sát.", "Chưa ghi là đã phỏng vấn."),
        ("Các nguồn dữ liệu khác nhãn và thời gian.", "Giữ nhãn gốc, ID người/phiên và timestamp trong manifest.",
         "Xin chốt bài toán phân lớp và tiêu chí cảnh báo.", ""),
    ],
    "next_title": "KẾ HOẠCH DỰ KIẾN TUẦN 2",
    "next": [
        ("Dữ liệu pilot", "Kiểm kê UTA crop, kiểm tra nhãn và chia train/validation/test theo người.",
         "Đầu tuần 2", "Luân và Huy"),
        ("Train baseline", "Trích feature mắt/miệng/đầu, train GRU camera và lưu checkpoint.", "Trong tuần 2", "Luân"),
        ("Kiểm thử model", "Đánh giá test khóa, kiểm tra runtime và báo cáo lỗi.", "Cuối tuần 2",
         "Luân; Huy rà kết quả"),
    ],
    "sources": [
        "Căn cứ: docs/reference/00_scope_mapping.md; docs/reference/01_architecture.md; docs/ROADMAP.md.",
        "Phân công theo vai trò trong bản báo cáo gốc; cần hai sinh viên xác nhận trước khi nộp.",
    ],
}

WEEK2 = {
    "subtitle": "Huấn luyện và kiểm thử baseline camera",
    "period": "Tuần 2",
    "goals": [
        "Huấn luyện GRU v1 trên UTA-RLDD crop pilot; lưu checkpoint và kiểm thử trên người không xuất hiện khi train.",
        "Đối chiếu CAM-01/CAM-02 và EEG độc lập để xác định giới hạn; kiểm tra luồng suy luận camera và cảnh báo.",
        "Rút ra nguyên nhân cần chuyển hướng model ở tuần 3: độ chính xác và khả năng khái quát còn thấp.",
    ],
    "tasks": [
        ("Dữ liệu camera pilot", "Chuẩn bị 354 clip 10 giây từ 59 người; chia theo người 39 train/10 validation/"
         "10 test.", "Huy; Luân phối hợp", "Đầu tuần 2", "Đã hoàn thành",
         "Corpus binary tỉnh/buồn ngủ, không trộn người giữa các tập.", "Nhãn theo clip, chưa có onset."),
        ("Feature và GRU v1", "Trích EAR, MAR, blendshape, yaw/pitch/roll và mask; train GRU với 50 mẫu quá khứ "
         "ở 10 Hz.", "Luân", "Giữa tuần 2", "Đã train", "Checkpoint camera-only và pipeline suy luận.",
         "Dùng validation chọn epoch/ngưỡng."),
        ("Test khóa GRU v1", "Đánh giá 10 người test, 60 clip; đo chỉ số theo clip và cửa sổ.",
         "Luân; Huy rà kết quả", "Giữa tuần 2", "Đã test",
         "Theo clip: accuracy/BA 55,00%; macro-F1 53,96%; AUROC 0,6444.",
         "18/30 clip tỉnh bị gọi buồn ngủ; chưa phải báo sai/giờ."),
        ("CAM-02 đa tín hiệu", "Train 4 cấu hình cue × 3 seed trên train/validation, lưu 12 checkpoint.", "Luân",
         "Cuối tuần 2", "Đã train development",
         "Full validation trung bình: accuracy 64,81%; macro-F1 64,11%; AUROC 0,6845.",
         "Chưa có test độc lập; không so trực tiếp với test v1."),
        ("EEG độc lập", "Đánh giá các baseline EEG v2/v3.1 như nhánh nghiên cứu độc lập.", "Luân; Huy tổng hợp",
         "Trong tuần 2", "Đã đánh giá", "EEG v2 test AUROC 0,6464, BA 50%; v3.1 LOSO AUROC 0,6494.",
         "Chưa đồng bộ video–EEG, chưa làm teacher."),
        ("Runtime và UI", "Kiểm tra suy luận video, quality gate, cảnh báo, API và dashboard.", "Luân và Huy",
         "Cuối tuần 2", "Prototype chạy được",
         "51 score video test khớp offline; API/dashboard và console debug đã có.",
         "P95 47,33 ms chỉ gồm perception+model+luật."),
        ("Kết luận tuần 2", "Phân tích lỗi và chốt kế hoạch cải tiến model.", "Luân và Huy", "Cuối tuần 2",
         "Đã tổng hợp", "Baseline đã train/test nhưng chưa đạt chất lượng kỳ vọng; chuyển sang dữ liệu dài và "
         "GRU+DANN.", "85% là mục tiêu, không phải kết quả."),
    ],
    "problems": [
        ("GRU v1 test còn nhiều nhầm lẫn.", "Rà nhãn, mở rộng dữ liệu dài, so model trên cùng split/protocol.",
         "Xin chốt tiêu chí chấp nhận model.", "Chưa kết luận dùng thực tế."),
        ("CAM-02 chỉ có validation.", "Giữ test khóa; dùng cohort mới để nghiệm thu ứng viên.",
         "Xin góp ý protocol kiểm chứng.", "Không chọn seed tốt nhất theo test."),
        ("EEG khác người/phiên với camera pilot.", "Chỉ ghép khi có dữ liệu camera–EEG đồng bộ như DROZY.",
         "Xin xác định vai trò EEG.", "Không coi EEG hiện có là ground truth camera."),
    ],
    "next_title": "KẾ HOẠCH DỰ KIẾN TUẦN 3",
    "next": [
        ("Audit dữ liệu mới", "Kiểm định UTA full, nhãn 0/5/10 và split theo người.", "Đầu tuần 3",
         "Điều kiện để train"),
        ("GRU và DANN", "Làm trainer ba lớp, domain head, resume và thí nghiệm đối chứng.", "Trong tuần 3",
         "Kế hoạch"),
        ("EEG supervision", "Chuẩn bị paired dataset, teacher và định nghĩa onset.", "Sau baseline camera",
         "Phụ thuộc dữ liệu đồng bộ"),
    ],
    "sources": [
        "Nguồn số liệu: docs/reports/12_camera_model_results.md; docs/reports/18_camera_multicue_results.md; "
        "docs/reports/09_model_v3_results.md.",
        "Các protocol test camera, validation CAM-02 và LOSO EEG khác nhau; không lấy chênh lệch làm mức cải thiện.",
    ],
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def _ms(entry: dict, key: str) -> str:
    """Mean ± between-seed SD, Vietnamese decimal comma."""
    value = entry[key]
    return f"{value['mean'] * 100:.1f} ± {value['sd'] * 100:.1f}%".replace(".", ",")


def _num(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _group(summary: dict, label_mode: str, mode: str) -> dict:
    return next(e for e in summary["aggregate"] if e["label_mode"] == label_mode and e["mode"] == mode)


def week3_content(pilot: dict, full: dict, first: dict, features: dict, tests: dict):
    """Week 3 sheet plus the explanation and evidence sheets."""
    pd, pg = _group(pilot, "binary_pilot", "dann"), _group(pilot, "binary_pilot", "gru")
    fd, fg = _group(full, "alert_drowsy", "dann"), _group(full, "alert_drowsy", "gru")
    td, tg = _group(full, "three_class", "dann"), _group(full, "three_class", "gru")
    v1 = _group(first, "alert_drowsy", "gru")
    secondary = full["secondary"]["alert_drowsy/120s/dann"]
    enrolled, ensemble = secondary["enrollment_ensemble"], secondary["ensemble"]
    ci = [run["window_ci"]["accuracy"] for run in full["runs"]
          if run["label_mode"] == "alert_drowsy" and run["mode"] == "dann"]
    ci_text = f"{min(c[0] for c in ci) * 100:.0f}–{max(c[1] for c in ci) * 100:.0f}%"
    pts = sum(v for k, v in features["timestamp_source"].items() if k.endswith("/pts"))
    passed = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    week = {
        "subtitle": "GRU+DANN trên video UTA-RLDD đầy đủ, sửa lỗi dữ liệu/DANN và chuẩn bị fine-tune EEG",
        "period": "Tuần 3 (đến 09/10/2026)",
        "goals": [
            "Huấn luyện và so sánh GRU với GRU+DANN theo cùng protocol, nhiều seed; báo cáo đúng tập validation.",
            "Hoàn tất kiểm định, trích feature UTA-RLDD đầy đủ và sửa các lỗi dữ liệu/huấn luyện phát hiện được.",
            "Có checkpoint camera ổn định và kế hoạch dữ liệu đồng bộ để bước sang fine-tune với EEG.",
        ],
        "tasks": [
            ("GRU+DANN pilot trên Kaggle",
             "Train GRU và GRU+DANN × 3 seed trên pilot 5 giây (binary) bằng Kaggle T4 ×2; chọn epoch theo validation.",
             "Luân", "06/10", "Đã train",
             f"Macro-F1 cửa sổ validation: DANN {_ms(pd, 'window_macro_f1')}, GRU {_ms(pg, 'window_macro_f1')}.",
             "DANN không tốt hơn GRU; overfit sau epoch 5–10."),
            ("Trích feature UTA full",
             f"Hash/probe 182 video, trích 19 feature 10 Hz; sửa lỗi {pts} video frame rate biến thiên "
             "(lệch PTS tới 1,8 s) bằng timestamp PTS của decoder.",
             "Huy; Luân phối hợp", "06–09/10", "Hoàn thành",
             f"{features['files']}/182 file, {features['subjects']} người, {_num(features['hours_at_10hz'], 2)} giờ; "
             f"lệch PTS tối đa {_num(features['max_pts_drift_ms'], 1)} ms.",
             "Feature test đã trích nhưng chưa mở."),
            ("Sửa lỗi DANN",
             "Phát hiện domain loss đứng ở ln(số người) — adversary không học. Tách learning rate domain head (×10), "
             "domain head nhận biểu diễn trước dropout.",
             "Luân", "09/10", "Đã sửa, có test",
             "Domain head học được; DANN có tác dụng khi encoder có tham số.", "Áp dụng cho mọi run sau 09/10."),
            ("Context dài, đặc trưng tương đối",
             "Tóm tắt từng giây (77 đặc trưng), độ lệch so với chính cửa sổ (PERCLOS/blink tương đối), context "
             "120 giây; kiến trúc window = thống kê cửa sổ + GRU khối 10 giây.",
             "Luân", "09/10", "Đã triển khai, có test",
             "Mô hình nhân quả; inference camera-only không cần EEG.", "Chọn qua 4 vòng phát triển trên validation."),
            ("UTA full alert vs drowsy",
             "GRU vs GRU+DANN × 3 seed, cấu hình khóa trong configs/training; chạy CPU local.",
             "Luân; Huy rà kết quả", "09/10", "Đã train",
             f"Accuracy cửa sổ validation: DANN {_ms(fd, 'window_accuracy')}, GRU {_ms(fg, 'window_accuracy')}; "
             f"AUROC {_num(fd['window_auroc_macro']['mean'], 3)}; làm mượt 5 phút {_ms(fd, 'smoothed_accuracy')}.",
             f"Vòng đầu chỉ {_ms(v1, 'window_accuracy')}. Chưa đạt >85%."),
            ("Ba lớp (tham chiếu)", "Cùng cấu hình, nhãn alert/low vigilance/drowsy.", "Luân", "09/10", "Đã train",
             f"Accuracy cửa sổ: DANN {_ms(td, 'window_accuracy')}, GRU {_ms(tg, 'window_accuracy')}; "
             f"macro-F1 ~{td['window_macro_f1']['mean'] * 100:.0f}%.",
             "Low vigilance khó tách; chưa dùng cho cảnh báo."),
            ("Hiệu chỉnh theo tài xế",
             "Phân tích phụ: 5 phút đầu coi như đoạn hiệu chỉnh đầu chuyến (giả định tỉnh), ngưỡng chọn "
             "leave-one-driver-out.",
             "Luân; Huy rà", "09/10", "Đã phân tích",
             f"Ensemble DANN: {_pct(enrolled['window_accuracy'])} cửa sổ, {_pct(enrolled['recording_accuracy'], 0)} "
             f"recording (không hiệu chỉnh: {_pct(ensemble['window_accuracy'])}).",
             "Giả định cần kiểm chứng; không phải kết quả chính."),
            ("Kiểm thử và đóng gói",
             "Pytest toàn bộ, ruff; notebook Kaggle cấu hình cuối; gói feature train/validation riêng.",
             "Luân", "09/10", "Đạt",
             f"{passed}/{tests['tests']} test đạt, {tests['skipped']} bỏ qua (cần CUDA), "
             f"{tests['failures'] + tests['errors']} lỗi.",
             "Chưa chạy lại cấu hình cuối trên Kaggle."),
            ("Bàn giao cho EEG",
             "Chốt checkpoint, biểu diễn 170 chiều và trạng thái DROZY cho kế hoạch fine-tune.",
             "Luân và Huy", "09/10", "Đã chuẩn bị",
             "Checkpoint DANN seed 42 (quy tắc chọn khai báo trước); hướng dẫn trong báo cáo 24.",
             "DROZY mới tải một phần."),
        ],
        "task_height": 78,
        "problems": [
            (f"Chưa đạt >85% ở đơn vị cửa sổ; CI theo người rộng ({ci_text}).",
             "Thêm dữ liệu (NTHU, DROZY), hiệu chỉnh theo tài xế, giám sát bằng EEG.",
             "Xin chốt đơn vị nghiệm thu (cửa sổ/làm mượt/recording) và nhiệm vụ nhị phân.",
             "Không ghi 85% là đã đạt."),
            ("Validation đã dùng để phát triển; test UTA full cùng người với test pilot cũ.",
             "Khóa model; mở test một lần cho cả camera-only và camera+EEG.",
             "Xin chốt thời điểm mở test và cohort kiểm chứng mới.", "Test chưa mở."),
            ("DROZY chưa tải đủ; lớp low vigilance khó tách.",
             "Tải/audit DROZY, căn chỉnh video–PSG trước khi train.",
             "Xin hỗ trợ quyền dữ liệu và định nghĩa onset/horizon.", "Chưa có kết quả cảnh báo sớm."),
        ],
        "next_title": "KẾ HOẠCH DỰ KIẾN TUẦN 4",
        "next": [
            ("Dữ liệu EEG đồng bộ", "Tải đủ, audit DROZY; căn chỉnh video NIR–PSG theo timestamp.", "Ưu tiên 1",
             "Điều kiện để fine-tune"),
            ("Teacher và fine-tune", "Teacher EEG theo người; student khởi tạo từ checkpoint GRU+DANN; ablation.",
             "Ưu tiên 2", "Cùng split/protocol"),
            ("Đánh giá khóa", "Lặp lại trên Kaggle T4 ×2; test UTA full một lần cho các model đã khóa.",
             "Cuối tuần 4", "Không tune bằng test"),
        ],
        "sources": [
            "Nguồn: docs/reports/23_gru_dann_kaggle_results_20261006.md; "
            "docs/reports/24_gru_dann_uta_full_20261009.md; outputs/audits/20261009/*.json.",
            "Pilot (clip 10 giây đã crop) và UTA full (video 10 phút) khác dữ liệu; không lấy chênh lệch làm "
            "mức cải thiện chính thức.",
            "Các mốc phân công và thời gian cần nhóm xác nhận trước khi nộp GVHD.",
        ],
    }
    explanation = [
        ("Đơn vị cửa sổ", "Một dự đoán cho 120 giây gần nhất (1.200 frame 10 Hz), mỗi 2,5 giây; chỉ dùng quá khứ. "
         "Accuracy/macro-F1 tính trên cửa sổ qua quality gate; coverage ghi tỉ lệ cửa sổ được chấp nhận."),
        ("Làm mượt 5 phút", "Trung bình xác suất các cửa sổ kết thúc trong 5 phút trước, cùng recording; dùng được "
         "online nhưng quyết định chậm hơn."),
        ("Recording", "Trung bình cả video ~10 phút; validation chỉ 20 recording nên mỗi lỗi = 5 điểm phần trăm."),
        ("Mean ± SD", "Trung bình và độ lệch chuẩn giữa 3 seed 42/43/44; không chọn seed tốt nhất."),
        ("CI 95%", "Bootstrap theo người (500 lần) trên validation; phản ánh dao động do chỉ có 10 tài xế."),
        ("Validation và test", "Validation (10 tài xế) dùng chọn epoch và đã dùng để phát triển kiến trúc qua 4 vòng, "
         "nên lạc quan. Test UTA full (10 tài xế) chưa mở."),
        ("alert vs drowsy", "UTA nhãn 0 → tỉnh, 10 → buồn ngủ; bỏ recording nhãn 5. Ba lớp được báo riêng."),
        ("DANN", "Domain head đoán người train qua gradient reversal để biểu diễn ít phụ thuộc người. Với kiến trúc "
         "cuối, domain head vẫn nhận ra người (thống kê cửa sổ không có tham số), nên DANN chỉ hơn GRU ~0,5 điểm."),
        ("Hiệu chỉnh theo tài xế", "Phân tích phụ, không phải kết quả chính: dùng 5 phút đầu recording tỉnh làm mức nền."),
        ("Lỗi VFR", "Một số video có frame rate thay đổi; timestamp index/fps lệch PTS thật tới 1,8 s. Nay dùng PTS "
         "của decoder cho các video này và ghi nguồn timestamp từng clip."),
        ("Mục tiêu 85%", "Accuracy hoặc macro-F1 cửa sổ > 0,85 trên đánh giá đúng protocol; hiện chưa đạt."),
    ]
    evidence = [
        ("MC01", "GRU v1 test pilot", "docs/results/camera_v1/summary.json; docs/reports/12_camera_model_results.md"),
        ("MC02", "Corpus pilot", "data/features/camera_uta_pilot/manifest.json"),
        ("MC03", "CAM-02", "docs/results/camera_cam02_development.json; docs/reports/18_camera_multicue_results.md"),
        ("MC04", "EEG độc lập", "docs/reports/06_model_results.md; docs/reports/09_model_v3_results.md"),
        ("MC05", "GRU+DANN pilot Kaggle", "models/from_kaggle/gru_dann_20261006/; "
         "outputs/audits/20261009/gru_dann_pilot_summary.json; docs/reports/23_gru_dann_kaggle_results_20261006.md"),
        ("MC06", "Feature UTA full v002", "data/features/uta_rldd_full_v002/manifest.json; "
         "outputs/audits/20261009/uta_full_v002_feature_audit.json"),
        ("MC07", "Sửa VFR/PTS", "driverguard/capture/source.py; driverguard/learning/camera/extract.py; "
         "tests/test_uta_full_validation.py"),
        ("MC08", "Trainer GRU+DANN, kiến trúc window", "driverguard/learning/camera/dann/; tests/test_camera_dann.py"),
        ("MC09", "Cấu hình khóa", "configs/training/camera_gru_dann_uta_full_v002_window.json"),
        ("MC10", "Run UTA full cuối", "runs/gru_dann/uta_full_v002_20261009_window/; "
         "outputs/audits/20261009/gru_dann_uta_full_window_summary.json"),
        ("MC11", "Các vòng phát triển", "runs/gru_dann/uta_full_v002_20261009/, _v2/; "
         "runs/gru_dann/dev_internal_split_20261009/; outputs/audits/20261009/validation_development/log.json"),
        ("MC12", "Kiểm thử", "outputs/audits/20261009/pytest.xml"),
        ("MC13", "Kaggle", "kaggle/03_train_gru_dann.py; outputs/kaggle_exports/20261009-uta-full/"),
        ("MC14", "Báo cáo chi tiết", "docs/reports/24_gru_dann_uta_full_20261009.md"),
    ]
    return week, explanation, evidence
