// Mirrors driverguard/schemas.py TickOutput.to_dict() — metadata only, never images.
export type Level = "normal" | "caution" | "warning" | "critical" | "sensor_degraded";

export interface Alert {
  ts_ms: number;
  level: Level;
  reasons: string[];
  message: string;
  score: number;
}

export interface Tick {
  timestamp_ms: number;
  trip_id: string;
  driver_id: string;
  camera_ok: boolean;
  face_valid: boolean;
  face_quality: number;
  calibrated: boolean;
  ear_left: number | null;
  ear_right: number | null;
  ear_norm: number | null;
  eye_closed: boolean | null;
  closure_ms: number;
  blink_duration_ms: number | null;
  blink_rate_per_min: number | null;
  perclos_proxy: number | null;
  mar: number | null;
  yawn_active: boolean;
  yawns_in_window: number;
  head_yaw_deg: number | null;
  head_pitch_deg: number | null;
  head_roll_deg: number | null;
  off_road_ms: number;
  phone_confidence: number;
  phone_persistence: number;
  speed_kmh: number | null;
  drive_time_min: number;
  fatigue_index: number | null;
  fatigue_trend: number | null;
  risk_score: number;
  risk_level: Level;
  reason_codes: string[];
  alert: Alert | null;
  processing_fps: number | null;
  latency_ms: number | null;
}

export interface Session {
  token: string;
  role: "driver" | "fleet_manager";
  username: string;
  driver_id: string | null;
}

export interface EventRow {
  id: number;
  trip_id: string;
  driver_id: string;
  ts_ms: number;
  wall_time: number;
  kind: string;
  level: Level;
  reasons: string[];
  score: number;
  message: string;
  feedback: string | null;
}

export interface Tunable {
  key: string;
  value: number;
  min: number | null;
  max: number | null;
  type: string;
}

export interface TripRow {
  trip_id: string;
  driver_id: string;
  started_at: number;
  ended_at: number | null;
  summary: Record<string, unknown> | null;
}
