"""Alert outputs (local only). Audio runs on a background thread so it never blocks inference."""
from __future__ import annotations

import sys
import threading

from ..schemas import AlertEvent, RiskLevel

_BEEPS = {  # (frequency Hz, duration ms, repeats)
    RiskLevel.CAUTION: (880, 150, 1),
    RiskLevel.WARNING: (1200, 250, 2),
    RiskLevel.CRITICAL: (1800, 350, 4),
    RiskLevel.SENSOR_DEGRADED: (500, 200, 1),
}


class ConsoleAlertSink:
    def on_alert(self, alert: AlertEvent) -> None:
        print(f"\n[ALERT {alert.level.value.upper()} @ {alert.ts_ms/1000:.1f}s] {alert.message}", flush=True)


class SoundAlertSink:
    """winsound on Windows; terminal bell elsewhere. TODO(android): vibration + TTS."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._busy = threading.Lock()

    def _play(self, level: RiskLevel) -> None:
        freq, dur, n = _BEEPS.get(level, (800, 150, 1))
        try:
            if sys.platform.startswith("win"):
                import winsound

                for _ in range(n):
                    winsound.Beep(freq, dur)
            else:
                for _ in range(n):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
        finally:
            self._busy.release()

    def on_alert(self, alert: AlertEvent) -> None:
        if not self.enabled or not self._busy.acquire(blocking=False):
            return  # drop if a sound is already playing
        threading.Thread(target=self._play, args=(alert.level,), daemon=True).start()
