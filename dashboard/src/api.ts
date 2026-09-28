import type { EventRow, Session, Tick, TripRow, Tunable } from "./types";

async function req<T>(path: string, session: Session | null, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (session) headers.Authorization = `Bearer ${session.token}`;
  const res = await fetch(path, { ...init, headers: { ...headers, ...(init.headers as object) } });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  login: (username: string, password: string) =>
    req<Session>("/auth/login", null, { method: "POST", body: JSON.stringify({ username, password }) }),
  state: (s: Session) => req<Tick>("/state", s),
  events: (s: Session, limit = 50) => req<EventRow[]>(`/events?limit=${limit}`, s),
  feedback: (s: Session, id: number, feedback: "correct" | "incorrect") =>
    req<{ ok: boolean }>(`/events/${id}/feedback`, s, { method: "POST", body: JSON.stringify({ feedback }) }),
  trips: (s: Session) => req<TripRow[]>("/trips", s),
  currentTrip: (s: Session) => req<Record<string, unknown>>("/trips/current/summary", s),
  config: (s: Session) => req<{ tunable: Tunable[]; calibration_method: string }>("/config", s),
  updateConfig: (s: Session, updates: Record<string, number>) =>
    req<{ ok: boolean; tunable: Tunable[] }>("/config", s, { method: "POST", body: JSON.stringify({ updates }) }),
};

/** Realtime metadata stream with auto-reconnect. Returns a close() function. */
export function openStream(
  session: Session,
  onTick: (t: Tick) => void,
  onStatus: (connected: boolean) => void,
): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry: number | undefined;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/stream?token=${encodeURIComponent(session.token)}`);
    ws.onopen = () => onStatus(true);
    ws.onmessage = (e) => onTick(JSON.parse(e.data) as Tick);
    ws.onclose = () => {
      onStatus(false);
      if (!closed) retry = window.setTimeout(connect, 1500);
    };
  };
  connect();
  return () => {
    closed = true;
    window.clearTimeout(retry);
    ws?.close();
  };
}
