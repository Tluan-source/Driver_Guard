"""Human-readable explanations (Vietnamese) from (level, reason codes).

ONE-WAY interface: explanation text is produced FROM the decision and is never fed back into
it. An optional SLM/LLM explainer may later implement the same signature
`(level, reasons) -> str`, receiving metadata only — never frames — and its output must not
change risk_level, thresholds or alerts (architecture rule, see docs/01_architecture.md).
"""
from __future__ import annotations

from ..schemas import RiskLevel

REASON_TEXT_VI = {
    "MICROSLEEP": "Mắt nhắm liên tục quá lâu (nghi ngủ gật)",
    "PROLONGED_EYE_CLOSURE": "Mắt nhắm lâu bất thường",
    "HIGH_PERCLOS": "Tỷ lệ nhắm mắt (PERCLOS-proxy) cao",
    "ELEVATED_PERCLOS": "Tỷ lệ nhắm mắt đang tăng",
    "FREQUENT_YAWN": "Ngáp nhiều lần",
    "HEAD_NOD": "Gật đầu (dấu hiệu buồn ngủ)",
    "FATIGUE_TREND_RISING": "Xu hướng mệt mỏi tăng dần trong chuyến",
    "RISK_SCORE": "Nhiều dấu hiệu rủi ro cộng dồn",
    "EYES_OFF_ROAD": "Không nhìn đường",
    "LOOKING_DOWN": "Cúi nhìn xuống lâu",
    "PHONE_USE": "Đang dùng điện thoại",
    "LONG_DRIVE": "Đã lái liên tục quá lâu",
    "CIRCADIAN_LOW": "Khung giờ dễ buồn ngủ",
    "PARKED": "Xe đang dừng",
    "CALIBRATING": "Đang hiệu chỉnh theo tài xế",
    "FACE_NOT_VISIBLE": "Không thấy khuôn mặt",
    "LOW_FACE_QUALITY": "Hình ảnh khuôn mặt kém (ánh sáng/che khuất)",
    "CAMERA_LOST": "Mất tín hiệu camera",
}

LEVEL_PREFIX_VI = {
    RiskLevel.CAUTION: "Chú ý",
    RiskLevel.WARNING: "CẢNH BÁO",
    RiskLevel.CRITICAL: "NGUY HIỂM",
    RiskLevel.SENSOR_DEGRADED: "Hệ thống không đánh giá được",
    RiskLevel.NORMAL: "Bình thường",
}

ADVICE_VI = {
    RiskLevel.CAUTION: "Hãy cân nhắc nghỉ ngơi ở điểm dừng gần nhất.",
    RiskLevel.WARNING: "Hãy tập trung và tìm chỗ an toàn để nghỉ.",
    RiskLevel.CRITICAL: "Dừng xe an toàn và nghỉ ngay!",
    RiskLevel.SENSOR_DEGRADED: "Kiểm tra camera / vị trí ngồi.",
    RiskLevel.NORMAL: "",
}


def explain_vi(level: RiskLevel, reasons: list[str]) -> str:
    parts = [REASON_TEXT_VI.get(r, r) for r in reasons if r not in ("CALIBRATING",)]
    body = "; ".join(parts) if parts else ""
    msg = LEVEL_PREFIX_VI.get(level, level.value)
    if body:
        msg += f": {body}."
    advice = ADVICE_VI.get(level, "")
    return f"{msg} {advice}".strip()
