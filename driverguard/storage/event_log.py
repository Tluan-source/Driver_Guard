"""Metadata-only event store (SQLite, local on the device).

Tables
  trips        — one row per trip (pseudonymous driver_id), summary JSON at the end
  events       — alerts + level transitions (+ driver feedback for "Tầng 3" value metrics)
  minute_stats — trip-memory buckets (fatigue index per minute)
  config_audit — every HITL threshold change: who, when, old -> new
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..privacy import assert_metadata_only

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id TEXT PRIMARY KEY, driver_id TEXT, started_at REAL, ended_at REAL, summary_json TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT, driver_id TEXT, ts_ms INTEGER, wall_time REAL,
    kind TEXT, level TEXT, reasons_json TEXT, score REAL, message TEXT, metrics_json TEXT,
    feedback TEXT, feedback_at REAL);
CREATE INDEX IF NOT EXISTS idx_events_trip ON events(trip_id, ts_ms);
CREATE TABLE IF NOT EXISTS minute_stats (
    trip_id TEXT, t_start_ms INTEGER, stats_json TEXT, PRIMARY KEY (trip_id, t_start_ms));
CREATE TABLE IF NOT EXISTS config_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, username TEXT, changes_json TEXT);
"""

_EVENT_METRICS = ("perclos_proxy", "closure_ms", "blink_rate_per_min", "yawns_in_window", "off_road_ms",
                  "phone_persistence", "speed_kmh", "drive_time_min", "fatigue_index", "face_quality")


class EventStore:
    def __init__(self, db_path: str | Path):
        db_path = str(db_path)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript(_SCHEMA)
            self._con.commit()

    # ------------------------------------------------------------------ writes
    def start_trip(self, trip_id: str, driver_id: str) -> None:
        with self._lock:
            self._con.execute("INSERT OR IGNORE INTO trips(trip_id, driver_id, started_at) VALUES (?,?,?)",
                              (trip_id, driver_id, time.time()))
            self._con.commit()

    def end_trip(self, trip_id: str, summary: dict[str, Any]) -> None:
        assert_metadata_only(summary)
        with self._lock:
            self._con.execute("UPDATE trips SET ended_at=?, summary_json=? WHERE trip_id=?",
                              (time.time(), json.dumps(summary), trip_id))
            self._con.commit()

    def log_event(self, tick: dict[str, Any], kind: str) -> int:
        """tick = TickOutput.to_dict(); stores the alert (or transition) + a few metrics."""
        alert = tick.get("alert") or {}
        metrics = {k: tick.get(k) for k in _EVENT_METRICS}
        row = {
            "trip_id": tick["trip_id"], "driver_id": tick["driver_id"], "ts_ms": tick["timestamp_ms"],
            "kind": kind, "level": alert.get("level", tick["risk_level"]),
            "reasons": alert.get("reasons", tick["reason_codes"]), "score": tick["risk_score"],
            "message": alert.get("message", ""), "metrics": metrics,
        }
        assert_metadata_only(row)
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO events(trip_id, driver_id, ts_ms, wall_time, kind, level, reasons_json, score, "
                "message, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (row["trip_id"], row["driver_id"], row["ts_ms"], time.time(), kind, row["level"],
                 json.dumps(row["reasons"]), row["score"], row["message"], json.dumps(metrics)))
            self._con.commit()
            return int(cur.lastrowid)

    def log_minute(self, trip_id: str, bucket: dict[str, Any]) -> None:
        assert_metadata_only(bucket)
        with self._lock:
            self._con.execute("INSERT OR REPLACE INTO minute_stats VALUES (?,?,?)",
                              (trip_id, bucket["t_start_ms"], json.dumps(bucket)))
            self._con.commit()

    def set_feedback(self, event_id: int, feedback: str) -> bool:
        if feedback not in ("correct", "incorrect"):
            raise ValueError("feedback must be 'correct' or 'incorrect'")
        with self._lock:
            cur = self._con.execute("UPDATE events SET feedback=?, feedback_at=? WHERE id=?",
                                    (feedback, time.time(), event_id))
            self._con.commit()
            return cur.rowcount > 0

    def audit_config(self, username: str, changes: dict[str, Any]) -> None:
        with self._lock:
            self._con.execute("INSERT INTO config_audit(at, username, changes_json) VALUES (?,?,?)",
                              (time.time(), username, json.dumps(changes)))
            self._con.commit()

    def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self._lock:
            n = self._con.execute("DELETE FROM events WHERE wall_time < ?", (cutoff,)).rowcount
            self._con.commit()
            return n

    # ------------------------------------------------------------------ reads
    def events(self, trip_id: str | None = None, driver_id: str | None = None, level: str | None = None,
               since_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        q = "SELECT * FROM events WHERE id > ?"
        args: list[Any] = [since_id]
        for col, val in (("trip_id", trip_id), ("driver_id", driver_id), ("level", level)):
            if val:
                q += f" AND {col} = ?"
                args.append(val)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["reasons"] = json.loads(d.pop("reasons_json") or "[]")
            d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
            out.append(d)
        return out

    def trips(self, driver_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM trips"
        args: list[Any] = []
        if driver_id:
            q += " WHERE driver_id = ?"
            args.append(driver_id)
        q += " ORDER BY started_at DESC"
        with self._lock:
            rows = self._con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json") or "null")
            out.append(d)
        return out

    def minute_stats(self, trip_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute("SELECT stats_json FROM minute_stats WHERE trip_id=? ORDER BY t_start_ms",
                                     (trip_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._con.close()
