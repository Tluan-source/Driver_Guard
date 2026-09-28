import { useState, type FormEvent } from "react";
import { api } from "../api";
import type { Session } from "../types";

export function Login({ onLogin }: { onLogin: (s: Session) => void }) {
  const [username, setUsername] = useState("manager");
  const [password, setPassword] = useState("manager");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLogin(await api.login(username, password));
    } catch {
      setError("Sai tên đăng nhập hoặc mật khẩu, hoặc backend chưa chạy.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="card login" onSubmit={submit}>
        <h1>DriverGuard</h1>
        <p className="muted">Giám sát tài xế on-device — dashboard chỉ nhận metadata, không có hình ảnh.</p>
        <label>
          Tài khoản
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
        </label>
        <label>
          Mật khẩu
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password" />
        </label>
        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={busy}>{busy ? "Đang đăng nhập…" : "Đăng nhập"}</button>
        <p className="muted small">Demo: <code>manager/manager</code> (quản lý đội xe) · <code>driver01/driver01</code> (tài xế)</p>
      </form>
    </div>
  );
}
