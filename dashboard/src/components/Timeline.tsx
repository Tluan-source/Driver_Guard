import type { Level, Tick } from "../types";

const LEVEL_VAR: Record<Level, string> = {
  normal: "var(--lv-normal)",
  caution: "var(--lv-caution)",
  warning: "var(--lv-warning)",
  critical: "var(--lv-critical)",
  sensor_degraded: "var(--lv-degraded)",
};

/** Last N seconds of risk score (line) + PERCLOS-proxy (dashed), with level-coloured bands. */
export function Timeline({ history, windowS = 300 }: { history: Tick[]; windowS?: number }) {
  const W = 900;
  const H = 180;
  const pad = { l: 34, r: 10, t: 10, b: 22 };
  if (history.length < 2) {
    return (
      <section className="card">
        <h2>Diễn biến 5 phút gần nhất</h2>
        <div className="muted">Chưa đủ dữ liệu</div>
      </section>
    );
  }
  const tEnd = history[history.length - 1].timestamp_ms;
  const tStart = tEnd - windowS * 1000;
  const pts = history.filter((t) => t.timestamp_ms >= tStart);
  const x = (ts: number) => pad.l + ((ts - tStart) / (windowS * 1000)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / 100) * (H - pad.t - pad.b);

  const bands: { x0: number; x1: number; level: Level }[] = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const last = bands[bands.length - 1];
    if (last && last.level === p.risk_level) last.x1 = x(p.timestamp_ms);
    else bands.push({ x0: x(p.timestamp_ms), x1: x(p.timestamp_ms), level: p.risk_level });
  }
  const score = pts.map((p) => `${x(p.timestamp_ms).toFixed(1)},${y(p.risk_score).toFixed(1)}`).join(" ");
  const perclos = pts
    .filter((p) => p.perclos_proxy != null)
    .map((p) => `${x(p.timestamp_ms).toFixed(1)},${y(Math.min(100, (p.perclos_proxy ?? 0) * 200)).toFixed(1)}`)
    .join(" ");

  return (
    <section className="card">
      <h2>Diễn biến 5 phút gần nhất</h2>
      <svg viewBox={`0 0 ${W} ${H}`} className="timeline" role="img"
        aria-label="Biểu đồ điểm rủi ro và PERCLOS-proxy theo thời gian">
        {bands.filter((b) => b.level !== "normal").map((b, i) => (
          <rect key={i} x={b.x0} y={pad.t} width={Math.max(2, b.x1 - b.x0)} height={H - pad.t - pad.b}
            fill={LEVEL_VAR[b.level]} opacity={0.18} />
        ))}
        {[0, 50, 100].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" className="axis">{v}</text>
          </g>
        ))}
        {[-300, -240, -180, -120, -60, 0].map((s) => (
          <text key={s} x={x(tEnd + s * 1000)} y={H - 6} className="axis"
            textAnchor={s === 0 ? "end" : s === -300 ? "start" : "middle"}>
            {s === 0 ? "bây giờ" : `${s / 60}p`}
          </text>
        ))}
        <polyline points={perclos} className="line-perclos" />
        <polyline points={score} className="line-score" />
      </svg>
      <div className="legend">
        <span><i className="sw sw-score" /> Điểm rủi ro</span>
        <span><i className="sw sw-perclos" /> PERCLOS-proxy ×200</span>
        <span><i className="sw sw-band" /> Nền màu = mức cảnh báo</span>
      </div>
    </section>
  );
}
