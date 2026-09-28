import type { Level } from "./types";

export const LEVEL_LABEL: Record<Level, string> = {
  normal: "Bình thường",
  caution: "Chú ý",
  warning: "Cảnh báo",
  critical: "Nguy hiểm",
  sensor_degraded: "Không đánh giá được",
};

export const REASON_LABEL: Record<string, string> = {
  MICROSLEEP: "Nhắm mắt quá lâu (nghi ngủ gật)",
  PROLONGED_EYE_CLOSURE: "Nhắm mắt lâu bất thường",
  HIGH_PERCLOS: "PERCLOS-proxy cao",
  ELEVATED_PERCLOS: "PERCLOS-proxy tăng",
  FREQUENT_YAWN: "Ngáp nhiều lần",
  HEAD_NOD: "Gật đầu",
  FATIGUE_TREND_RISING: "Xu hướng mệt tăng",
  RISK_SCORE: "Nhiều dấu hiệu cộng dồn",
  EYES_OFF_ROAD: "Không nhìn đường",
  LOOKING_DOWN: "Cúi nhìn xuống",
  PHONE_USE: "Dùng điện thoại",
  LONG_DRIVE: "Lái liên tục quá lâu",
  CIRCADIAN_LOW: "Khung giờ dễ buồn ngủ",
  PARKED: "Xe đang dừng",
  CALIBRATING: "Đang hiệu chỉnh",
  FACE_NOT_VISIBLE: "Không thấy mặt",
  LOW_FACE_QUALITY: "Ảnh mặt kém",
  CAMERA_LOST: "Mất camera",
};

export const TUNABLE_LABEL: Record<string, string> = {
  "eye.closed_ratio_enter": "Ngưỡng nhắm mắt (EAR / baseline)",
  "eye.long_closure_ms": "Nhắm lâu → WARNING (ms)",
  "eye.microsleep_ms": "Nhắm lâu → CRITICAL (ms)",
  "yawn.min_duration_ms": "Thời lượng tối thiểu của cái ngáp (ms)",
  "head.distraction_ms": "Nhìn lệch → WARNING (ms)",
  "phone.min_confidence": "Độ tin cậy tối thiểu phát hiện điện thoại",
  "risk.perclos_caution": "PERCLOS-proxy → CAUTION",
  "risk.perclos_warning": "PERCLOS-proxy → WARNING",
  "risk.alert_cooldown_ms": "Khoảng lặp cảnh báo (ms)",
};

export const fmt = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(digits);
