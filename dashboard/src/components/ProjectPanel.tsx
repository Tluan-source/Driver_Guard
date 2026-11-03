import { useCallback, useEffect, useState } from "react";
import type { Session } from "../types";

type Metrics = { accuracy: number | null; macro_f1: number | null; coverage: number | null };
type Run = {
  id: string; name: string; family: string; seed: number | null;
  validation: Metrics; cross_validation: boolean; outer_fold: number | null;
  eligible_for_selection: boolean; status: string;
};
type Ensemble = { window_accuracy: number; window_macro_f1: number; coverage: number };
type Overview = {
  target: { achieved: boolean; candidate: string | null; threshold: number };
  runs: Run[];
  stages: { id: string; title: string; state: string; next: string }[];
  datasets: { id: string; name: string; file_count: number; records: number; status: string }[];
  development_cv: {
    run_count: number; artifact_audit: string;
    models: Record<string, Ensemble> | null; candidate_mode: string | null;
  };
};
const pct = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;

export function ProjectPanel({ session }: { session: Session }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const response = await fetch("/project/overview", { headers: { Authorization: `Bearer ${session.token}` } });
      if (!response.ok) throw new Error(`Không đọc được tổng quan (${response.status}).`);
      setOverview(await response.json() as Overview); setError("");
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }, [session.token]);
  useEffect(() => { void refresh(); }, [refresh]);
  return <section className="card project-panel">
    <div className="project-heading"><div><h2>Tiến độ mô hình · Tuần 5</h2>
      <p className="muted small">Mục tiêu accuracy &gt;85% trên cửa sổ 120 giây; báo kèm macro-F1 và coverage.</p></div>
      <button className="secondary" disabled={busy} onClick={() => void refresh()}>{busy ? "Đang đọc…" : "Tải lại"}</button>
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {overview && <div className="stack">
      <p role="status">{overview.target.achieved ? "Đã có kết quả test độc lập đạt mục tiêu." : "Chưa xác nhận mục tiêu bằng test độc lập."}{" "}
        {overview.target.candidate ? `Ứng viên: ${overview.target.candidate}` : "Chưa có checkpoint cuối đủ điều kiện chọn."}</p>
      {overview.development_cv.models && <div className="table-scroll"><table>
        <caption>Ensemble 3 seed · 5 fold development · {overview.development_cv.run_count} lượt train</caption>
        <thead><tr><th>Mô hình</th><th>Accuracy</th><th>Macro-F1</th><th>Coverage</th></tr></thead>
        <tbody>{Object.entries(overview.development_cv.models).map(([name, row]) => <tr key={name}>
          <td>{name === "dann" ? "GRU+DANN" : "GRU"}</td><td>{pct(row.window_accuracy)}</td>
          <td>{pct(row.window_macro_f1)}</td><td>{pct(row.coverage)}</td>
        </tr>)}</tbody>
      </table></div>}
      <p className="muted small">CV dùng để so sánh hướng phát triển; các checkpoint từng fold được loại khỏi danh sách chọn model cuối.
        Kiểm định artifact: {overview.development_cv.artifact_audit}.</p>
      <div><a href="http://127.0.0.1:8010" target="_blank" rel="noreferrer">Mở Model Lab cục bộ ↗</a>
        <p className="muted small">Chạy <code>python -m driverguard debug-ui</code> để replay feature hoặc kiểm tra tín hiệu trực tiếp.</p></div>
      <details><summary>{overview.runs.length} lượt train có metadata</summary><div className="table-scroll"><table>
        <thead><tr><th>Run</th><th>Phạm vi</th><th>Inner accuracy</th><th>Inner macro-F1</th><th>Chọn cuối</th></tr></thead>
        <tbody>{overview.runs.map(run => <tr key={run.id}><td>{run.name}</td>
          <td>{run.cross_validation ? `CV fold ${run.outer_fold}` : run.status}</td>
          <td>{pct(run.validation.accuracy)}</td><td>{pct(run.validation.macro_f1)}</td>
          <td>{run.eligible_for_selection ? "Ứng viên development" : "Loại khỏi lựa chọn"}</td></tr>)}</tbody>
      </table></div></details>
      <details><summary>Dữ liệu trên máy</summary><ul>{overview.datasets.map(item => <li key={item.id}>
        {item.name}: {item.records} clip, {item.file_count} file · {item.status}
      </li>)}</ul></details>
      <details><summary>Tiến độ và phần còn thiếu</summary><ul>{overview.stages.map(stage => <li key={stage.id}>
        <strong>{stage.title}</strong> · {stage.state}<p className="muted small">{stage.next}</p>
      </li>)}</ul></details>
    </div>}
  </section>;
}
