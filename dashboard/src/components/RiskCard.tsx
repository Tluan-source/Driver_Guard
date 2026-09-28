import { LEVEL_LABEL, REASON_LABEL } from "../labels";
import type { Tick } from "../types";

export function RiskCard({ tick }: { tick: Tick | null }) {
  if (!tick) {
    return (
      <section className="card risk level-none">
        <div className="risk-level">Đang chờ dữ liệu…</div>
      </section>
    );
  }
  const reasons = tick.reason_codes.filter((r) => r !== "CALIBRATING");
  return (
    <section className={`card risk level-${tick.risk_level}`} aria-live="polite">
      <div className="risk-top">
        <div>
          <div className="eyebrow">Mức rủi ro hiện tại</div>
          <div className="risk-level">{LEVEL_LABEL[tick.risk_level]}</div>
        </div>
        <div className="score" title="Điểm rủi ro tổng hợp (0–100) — chỉ để hiển thị; quyết định do state machine">
          <span className="score-num">{Math.round(tick.risk_score)}</span>
          <span className="muted small">/100</span>
        </div>
      </div>
      <div className="chips">
        {reasons.length === 0 && <span className="muted">Không có dấu hiệu bất thường</span>}
        {reasons.map((r) => (
          <span key={r} className="chip">{REASON_LABEL[r] ?? r}</span>
        ))}
      </div>
      {!tick.calibrated && (
        <div className="calib">Đang hiệu chỉnh theo tài xế (60 giây đầu chuyến) — dùng ngưỡng mặc định</div>
      )}
      {tick.alert && <div className="alert-msg">{tick.alert.message}</div>}
    </section>
  );
}
