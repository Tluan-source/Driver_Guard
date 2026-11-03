# %% [markdown]
# # DriverGuard — kiểm kê dữ liệu cho GRU + DANN → EEG
#
# Notebook chuẩn bị dữ liệu, **không huấn luyện model**. Chỉ Add Input các bộ
# đã có quyền xử lý trên Kaggle. Giữ notebook/output Private nếu điều khoản yêu cầu.
# Đọc `docs/guides/19_datasets_gru_dann_eeg_kaggle.md` để chọn bộ và hiểu giới hạn nhãn.
# Không upload nguyên folder data của dự án vì mỗi bộ có quyền sử dụng khác nhau.
#
# Input chỉ đọc ở `/kaggle/input`; output ở `/kaggle/working/dataset_intake`.
# Không cần Internet, không cài dependency, không tải dữ liệu từ nguồn khác.

# %%
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/dataset_intake")
if not INPUT_ROOT.is_dir():
    raise RuntimeError("Chạy trên Kaggle và Add Input dataset trước; notebook không chạy train local.")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Input mounts:", [p.name for p in INPUT_ROOT.iterdir()])

# %% [markdown]
# GPU chỉ cần cho huấn luyện sau này; kiểm kê/trích MediaPipe có thể dùng CPU.
# Phép nhân ma trận dưới đây chỉ kiểm tra CUDA; không phải training/benchmark.
# Bật GPU trong Settings không tự chuyển trainer CPU cũ sang CUDA.

# %%
import json

import torch

runtime = {"torch_version": str(torch.__version__), "cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    runtime["gpu"] = torch.cuda.get_device_name(0)
    probe = torch.ones((16, 16), device="cuda")
    assert float((probe @ probe)[0, 0]) == 16.0
    del probe
print(runtime)
(OUTPUT_ROOT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

# %% [markdown]
# Kiểm kê chỉ đọc tên/kích thước, không giải mã video hoặc nạp toàn bộ EEG.
# `mount_group` chỉ là nhóm mount, **không phải** domain/camera/subject đáng tin cậy.
# Chuẩn hóa dataset_id và ID người/phiên theo tài liệu nguồn ở bước adapter sau.

# %%
import csv
from collections import Counter


def inventory(input_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    counts, byte_counts = Counter(), Counter()
    total_bytes = 0
    with (output_root / "asset_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "mount_group", "extension", "bytes"])
        writer.writeheader()
        for path in sorted(input_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(input_root)
            size = path.stat().st_size
            extension = path.suffix.lower() or "(none)"
            writer.writerow({"relative_path": relative.as_posix(), "mount_group": relative.parts[0],
                             "extension": extension, "bytes": size})
            counts[extension] += 1
            byte_counts[extension] += size
            total_bytes += size
    result = {"files": sum(counts.values()), "bytes": total_bytes,
              "counts_by_extension": dict(counts), "bytes_by_extension": dict(byte_counts)}
    (output_root / "inventory_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


print(json.dumps(inventory(INPUT_ROOT, OUTPUT_ROOT), indent=2))

# %% [markdown]
# Template cho adapter dữ liệu: một hàng/recording, namespace ID người theo
# dataset. Không suy nhãn từ model hoặc coi mọi EEG độc lập là đồng bộ với video.
# camera_id/illumination không rõ phải là unknown; không đoán từ danh tính người.
# Chia subject/session trước cắt cửa sổ/augmentation. Final test không tham gia
# DANN adaptation nếu báo cáo là generalization sang miền/người chưa thấy.
# Template dưới đây chỉ có header, chưa phải manifest hợp lệ để training.

# %%
recording_columns = [
    "dataset_id", "subject_id", "session_id", "source_recording_id", "camera_id",
    "illumination", "modality", "video_path", "eeg_path", "timestamps_path",
    "label_source", "label_path", "sampling_rate", "split", "sha256",
]
with (OUTPUT_ROOT / "recording_manifest_template.csv").open("w", newline="", encoding="utf-8") as stream:
    csv.writer(stream).writerow(recording_columns)
print("Saved intake outputs to", OUTPUT_ROOT)

# %% [markdown]
# Sau kiểm kê: viết adapter giữ timestamp/mask, khóa split, rồi triển khai
# baseline GRU và GRU+DANN CUDA trên cùng protocol. Trainer GRU+DANN chưa có
# trong notebook này. Save Version lưu output; tải bảng inventory về để chốt
# cấu trúc file trước khi trích feature/training. Không cần tải video về laptop.
