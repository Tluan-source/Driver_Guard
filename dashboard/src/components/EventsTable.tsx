import { LEVEL_LABEL, REASON_LABEL } from "../labels";
import type { EventRow, Session } from "../types";

export function EventsTable({ events, session, onFeedback }: {
  events: EventRow[];
  session: Session;
  onFeedback: (id: number, fb: "correct" | "incorrect") => void;
}) {
  const isDriver = session.role === "driver";
  return (
    <section className="card">
      <h2>Nhật ký sự kiện</h2>
      {events.length === 0 ? (
        <div className="muted">Chưa có sự kiện</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Thời gian</th>
                {!isDriver && <th>Tài xế</th>}
                <th>Mức</th>
                <th>Lý do</th>
                <th>Phản hồi</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id}>
                  <td className="mono">{new Date(e.wall_time * 1000).toLocaleTimeString("vi-VN")}</td>
                  {!isDriver && <td className="mono">{e.driver_id}</td>}
                  <td><span className={`pill level-${e.level}`}>{LEVEL_LABEL[e.level] ?? e.level}</span></td>
                  <td>{e.reasons.filter((r) => r !== "CALIBRATING").map((r) => REASON_LABEL[r] ?? r).join(", ")}</td>
                  <td>
                    {e.kind !== "alert" ? (
                      <span className="muted small">chuyển mức</span>
                    ) : e.feedback ? (
                      <span className="muted">{e.feedback === "correct" ? "Đúng" : "Sai"}</span>
                    ) : isDriver ? (
                      <span className="fb">
                        <button onClick={() => onFeedback(e.id, "correct")}>Đúng</button>
                        <button className="secondary" onClick={() => onFeedback(e.id, "incorrect")}>Sai</button>
                      </span>
                    ) : (
                      <span className="muted small">chờ tài xế</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
