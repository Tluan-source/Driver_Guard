"""Privacy invariant: only METADATA may leave the perception process.

`assert_metadata_only` is called on every payload before it is stored (SQLite), published
(API / WebSocket) or written to JSONL. It rejects binary data, arrays, very long strings
(e.g. base64 images) and keys that look like image / landmark / biometric payloads.

This is a code-level guard; the acceptance test is still a packet capture showing 0 image bytes
off-device (docs/01_architecture.md, "Privacy test").
"""
from __future__ import annotations

from typing import Any

FORBIDDEN_KEY_PARTS = ("image", "frame_b", "jpeg", "jpg", "png", "pixel", "landmark", "video",
                       "face_crop", "embedding", "base64", "thumbnail", "snapshot_img")
MAX_STR_LEN = 512
MAX_LIST_LEN = 20000


class PrivacyViolation(ValueError):
    pass


def assert_metadata_only(obj: Any, path: str = "$") -> None:
    if obj is None or isinstance(obj, (bool, int, float)):
        return
    if isinstance(obj, str):
        if len(obj) > MAX_STR_LEN:
            raise PrivacyViolation(f"{path}: string too long ({len(obj)} chars) — possible encoded image")
        return
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raise PrivacyViolation(f"{path}: binary payload is not allowed")
    if hasattr(obj, "shape") and getattr(obj, "ndim", 0) > 0:
        raise PrivacyViolation(f"{path}: array payload is not allowed")
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(part in lk for part in FORBIDDEN_KEY_PARTS):
                raise PrivacyViolation(f"{path}.{k}: forbidden key")
            assert_metadata_only(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST_LEN:
            raise PrivacyViolation(f"{path}: list too long ({len(obj)})")
        for i, v in enumerate(obj):
            assert_metadata_only(v, f"{path}[{i}]")
        return
    if hasattr(obj, "item"):  # numpy scalar
        return
    raise PrivacyViolation(f"{path}: unsupported type {type(obj).__name__}")
