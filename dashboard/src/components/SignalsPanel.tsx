import { fmt } from "../labels";
import type { Tick } from "../types";

function Stat({ label, value, hint, warn }: { label: string; value: string; hint?: string; warn?: boolean }) {
  return (
    <div className={`stat ${warn ? "warn" : ""}`} title={hint}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

export function SignalsPanel({ tick }: { tick: Tick | null }) {
  const t = tick;
  return (
    <section className="card">
      <h2>Tín hiệu</h2>
      <div className="stats">
        <Stat label="PERCLOS-proxy (60s)" value={fmt(t?.perclos_proxy)} warn={(t?.perclos_proxy ?? 0) >= 0.15}
          hint="Tỷ lệ thời gian nhắm mắt ước lượng từ EAR — xấp xỉ, không phải PERCLOS đo bằng IR" />
        <Stat label="EAR / baseline" value={fmt(t?.ear_norm)} hint="EAR chia cho mức mắt mở cá nhân" />
        <Stat label="Nhắm hiện tại" value={t ? `${t.closure_ms} ms` : "–"} warn={(t?.closure_ms ?? 0) >= 1000} />
        <Stat label="Chớp mắt / phút" value={fmt(t?.blink_rate_per_min, 1)} />
        <Stat label="Ngáp (5 phút)" value={t ? String(t.yawns_in_window) : "–"} warn={(t?.yawns_in_window ?? 0) >= 2} />
        <Stat label="Yaw / Pitch" value={t ? `${fmt(t.head_yaw_deg, 0)}° / ${fmt(t.head_pitch_deg, 0)}°` : "–"}
          hint="So với tư thế trung tính đã hiệu chỉnh" />
        <Stat label="Điện thoại" value={fmt(t?.phone_persistence)} warn={(t?.phone_persistence ?? 0) >= 0.6} />
        <Stat label="Chất lượng mặt" value={fmt(t?.face_quality)} warn={t ? !t.face_valid : false} />
        <Stat label="Tốc độ (CAN giả lập)" value={t?.speed_kmh != null ? `${fmt(t.speed_kmh, 0)} km/h` : "–"} />
        <Stat label="Lái liên tục" value={t ? `${fmt(t.drive_time_min, 0)} phút` : "–"} />
        <Stat label="Fatigue index" value={fmt(t?.fatigue_index, 0)}
          hint="Chỉ số mệt theo phút (công thức seed, cần kiểm chứng)" />
        <Stat label="Xu hướng mệt" value={t?.fatigue_trend != null ? `${fmt(t.fatigue_trend, 2)}/phút` : "–"}
          warn={(t?.fatigue_trend ?? 0) >= 0.5} />
      </div>
    </section>
  );
}
