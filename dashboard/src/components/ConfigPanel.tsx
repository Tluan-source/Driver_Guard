import { useEffect, useState } from "react";
import { api } from "../api";
import { TUNABLE_LABEL } from "../labels";
import type { Session, Tunable } from "../types";

/** HITL threshold tuning — fleet_manager only; every change is audited on the backend. */
export function ConfigPanel({ session }: { session: Session }) {
  const [items, setItems] = useState<Tunable[]>([]);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api.config(session).then((c) => setItems(c.tunable)).catch((e) => setMsg(String(e)));
  }, [session]);

  const changed = Object.entries(draft).filter(([k, v]) => {
    const cur = items.find((i) => i.key === k);
    return cur && v !== "" && Number(v) !== cur.value;
  });

  const save = async () => {
    const updates = Object.fromEntries(changed.map(([k, v]) => [k, Number(v)]));
    try {
      const r = await api.updateConfig(session, updates);
      setItems(r.tunable);
      setDraft({});
      setMsg("Đã áp dụng (có ghi nhật ký thay đổi).");
    } catch (e) {
      setMsg(`Không hợp lệ: ${String(e).slice(0, 160)}`);
    }
  };

  return (
    <section className="card">
      <h2>Hiệu chỉnh ngưỡng (HITL)</h2>
      <p className="muted small">Chỉ quản lý đội xe được chỉnh. Ngưỡng nên chọn trên tập validation, không trên tập test.</p>
      <div className="cfg">
        {items.map((i) => (
          <label key={i.key} className="cfg-row">
            <span>{TUNABLE_LABEL[i.key] ?? i.key}</span>
            <input type="number" step={i.type === "int" ? 50 : 0.01} min={i.min ?? undefined} max={i.max ?? undefined}
              value={draft[i.key] ?? String(i.value)}
              onChange={(e) => setDraft({ ...draft, [i.key]: e.target.value })} />
            <span className="muted small">{i.min ?? "–"} … {i.max ?? "–"}</span>
          </label>
        ))}
      </div>
      <div className="row">
        <button onClick={save} disabled={changed.length === 0}>Áp dụng {changed.length ? `(${changed.length})` : ""}</button>
        {msg && <span className="muted small">{msg}</span>}
      </div>
    </section>
  );
}
