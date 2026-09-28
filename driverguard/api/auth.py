"""Minimal auth for the two roles required by the brief: driver and fleet_manager.

Pure standard library (PBKDF2 + HMAC-signed tokens) so it is testable without FastAPI.
Data-access policy (docs/01_architecture.md):
  * driver         — sees only their own live state, events and trips; can give alert feedback.
  * fleet_manager  — sees all drivers' METADATA (never images); can tune HITL thresholds.

CLI:  python -m driverguard.api.auth hash "new-password"
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ROLES = ("driver", "fleet_manager")


def hash_password(password: str, salt: bytes | None = None, iterations: int = 200_000) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
    return hmac.compare_digest(dk.hex(), hash_hex)


@dataclass
class User:
    username: str
    role: str
    driver_id: str | None = None
    password_hash: str | None = None
    password: str | None = None  # plaintext allowed ONLY in the example/demo file

    def check(self, pw: str) -> bool:
        if self.password_hash:
            return verify_password(pw, self.password_hash)
        return self.password is not None and hmac.compare_digest(self.password, pw)


@dataclass
class Principal:
    username: str
    role: str
    driver_id: str | None

    @property
    def is_manager(self) -> bool:
        return self.role == "fleet_manager"


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class AuthService:
    def __init__(self, users: list[User], secret: bytes | None = None, ttl_s: int = 43200):
        for u in users:
            if u.role not in ROLES:
                raise ValueError(f"unknown role {u.role!r} for {u.username}")
        self.users = {u.username: u for u in users}
        env = os.environ.get("DRIVERGUARD_SECRET")
        self.secret = secret or (env.encode() if env else secrets.token_bytes(32))
        self.ttl_s = ttl_s

    @classmethod
    def from_file(cls, path: str | Path, ttl_s: int = 43200) -> "AuthService":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls([User(**u) for u in data.get("users", [])], ttl_s=ttl_s)

    def _sign(self, payload: bytes) -> str:
        return _b64(hmac.new(self.secret, payload, hashlib.sha256).digest())

    def login(self, username: str, password: str) -> str | None:
        u = self.users.get(username)
        if u is None or not u.check(password):
            return None
        payload = json.dumps({"u": u.username, "r": u.role, "d": u.driver_id,
                              "exp": int(time.time()) + self.ttl_s}).encode()
        return f"{_b64(payload)}.{self._sign(payload)}"

    def verify(self, token: str | None) -> Principal | None:
        if not token or "." not in token:
            return None
        p64, sig = token.rsplit(".", 1)
        try:
            payload = _unb64(p64)
        except Exception:
            return None
        if not hmac.compare_digest(self._sign(payload), sig):
            return None
        data = json.loads(payload)
        if data.get("exp", 0) < time.time():
            return None
        return Principal(data["u"], data["r"], data.get("d"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "hash":
        print(hash_password(sys.argv[2]))
    else:
        print('usage: python -m driverguard.api.auth hash "password"')
