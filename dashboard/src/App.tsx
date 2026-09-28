import { useCallback, useEffect, useRef, useState } from "react";
import { api, openStream } from "./api";
import { ConfigPanel } from "./components/ConfigPanel";
import { EventsTable } from "./components/EventsTable";
import { Login } from "./components/Login";
import { RiskCard } from "./components/RiskCard";
import { SignalsPanel } from "./components/SignalsPanel";
import { Timeline } from "./components/Timeline";
import { TripPanel } from "./components/TripPanel";
import { fmt } from "./labels";
import type { EventRow, Session, Tick } from "./types";

const HISTORY_MS = 5 * 60 * 1000;

function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem("dg_session");
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [tick, setTick] = useState<Tick | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<EventRow[]>([]);
  const history = useRef<Tick[]>([]);
  const [, force] = useState(0);

  const login = (s: Session) => {
    try {
      sessionStorage.setItem("dg_session", JSON.stringify(s));
    } catch {
      /* private mode: keep in memory only */
    }
    setSession(s);
  };
  const logout = () => {
    try {
      sessionStorage.removeItem("dg_session");
    } catch {
      /* ignore */
    }
    setSession(null);
    history.current = [];
    setTick(null);
  };

  useEffect(() => {
    if (!session) return;
    let frame = 0;
    const close = openStream(
      session,
      (t) => {
        const h = history.current;
        if (h.length && t.timestamp_ms < h[h.length - 1].timestamp_ms) h.length = 0; // new trip / replay restart
        h.push(t);
        while (h.length && h[0].timestamp_ms < t.timestamp_ms - HISTORY_MS) h.shift();
        setTick(t);
        if (++frame % 5 === 0) force((x) => x + 1);
      },
      setConnected,
    );
    return close;
  }, [session]);

  const refreshEvents = useCallback(() => {
    if (!session) return;
    api.events(session, 50).then(setEvents).catch((e) => {
      if (String(e).startsWith("Error: 401")) logout();
    });
  }, [session]);

  useEffect(() => {
    refreshEvents();
    const id = window.setInterval(refreshEvents, 5000);
    return () => window.clearInterval(id);
  }, [refreshEvents]);

  useEffect(() => {
    if (tick?.alert) refreshEvents();
  }, [tick?.alert, refreshEvents]);

  if (!session) return <Login onLogin={login} />;

  const isManager = session.role === "fleet_manager";
  return (
    <div className="app">
      <header>
        <div className="brand">
          <strong>DriverGuard</strong>
          <span className="muted small">metadata-only · không truyền hình ảnh khuôn mặt</span>
        </div>
        <div className="hdr-right">
          <span className={`dot ${connected ? "on" : "off"}`} />
          <span className="small">{connected ? "Đang nhận dữ liệu" : "Mất kết nối"}</span>
          <span className="small mono">{fmt(tick?.processing_fps, 1)} FPS · {fmt(tick?.latency_ms, 0)} ms</span>
          <span className="small">
            {session.username} ({isManager ? "Quản lý đội xe" : `Tài xế ${session.driver_id ?? ""}`})
          </span>
          <button className="secondary" onClick={logout}>Đăng xuất</button>
        </div>
      </header>
      <main>
        <div className="grid-top">
          <RiskCard tick={tick} />
          <SignalsPanel tick={tick} />
        </div>
        <Timeline history={history.current} />
        <div className="grid-bottom">
          <EventsTable
            events={events}
            session={session}
            onFeedback={(id, fb) => api.feedback(session, id, fb).then(refreshEvents)}
          />
          <div className="stack">
            <TripPanel session={session} />
            {isManager && <ConfigPanel session={session} />}
          </div>
        </div>
        <p className="muted small foot">
          Camera preview chỉ hiển thị cục bộ trên thiết bị (<code>driverguard run --show</code>); dashboard này không
          bao giờ nhận ảnh. Các ngưỡng là tham số kỹ thuật, không phải chuẩn y khoa.
        </p>
      </main>
    </div>
  );
}
