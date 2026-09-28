import { useEffect, useState } from "react";
import { api } from "../api";
import { fmt } from "../labels";
import type { Session } from "../types";

interface Summary {
  trip_id?: string;
  driver_id?: string;
  minutes?: number;
  degraded_ratio?: number;
  alerts?: Record<string, number>;
  alerts_per_hour?: number | null;
  long_closures?: number;
  yawns?: number;
  fatigue_index_last?: number | null;
  fatigue_trend_per_min?: number | null;
  fatigue_series?: [number, number | null][];
}

/** Trip-level record: what the fleet manager sees (metadata only — never the driver's face). */
export function TripPanel({ session }: { session: Session }) {
  const [s, setS] = useState<Summary | null>(null);

  useEffect(() => {
    const load = () => api.currentTrip(session).then((x) => setS(x as Summary)).catch(() => undefined);
    load();
    const id = window.setInterval(load, 10000);
    return () => window.clearInterval(id);
  }, [session]);

  const series = (s?.fatigue_series ?? []).filter((p) => p[1] != null) as [number, number][];
  const W = 280;
  const H = 60;
  const path = series
    .map((p, i) => `${(i / Math.max(1, series.length - 1)) * W},${H - (p[1] / 100) * H}`)
    .join(" ");

  return (
    <section className="card">
      <h2>Chuyến hiện tại</h2>
      {!s ? (
        <div className="muted">Đang tải…</div>
      ) : (
        <>
          <div className="stats three">
            <div className="stat"><div className="stat-label">Thời lượng</div><div className="stat-value">{fmt(s.minutes, 0)} phút</div></div>
            <div className="stat"><div className="stat-label">Cảnh báo / giờ</div><div className="stat-value">{fmt(s.alerts_per_hour ?? null, 1)}</div></div>
            <div className="stat" title="Tỷ lệ thời gian hệ thống không đánh giá được — cũng là chỉ báo camera bị che">
              <div className="stat-label">Tỷ lệ degraded</div>
              <div className="stat-value">{s.degraded_ratio != null ? `${(s.degraded_ratio * 100).toFixed(1)}%` : "–"}</div>
            </div>
            <div className="stat"><div className="stat-label">Nhắm mắt lâu</div><div className="stat-value">{s.long_closures ?? 0}</div></div>
            <div className="stat"><div className="stat-label">Ngáp</div><div className="stat-value">{s.yawns ?? 0}</div></div>
            <div className="stat"><div className="stat-label">Xu hướng mệt</div><div className="stat-value">{fmt(s.fatigue_trend_per_min ?? null, 2)}</div></div>
          </div>
          <div className="spark-wrap">
            <div className="muted small">Fatigue index theo phút</div>
            {series.length > 1 ? (
              <svg viewBox={`0 0 ${W} ${H}`} className="spark" aria-label="Fatigue index theo phút">
                <polyline points={path} className="line-score" />
              </svg>
            ) : (
              <div className="muted small">Cần ≥ 2 phút dữ liệu</div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
