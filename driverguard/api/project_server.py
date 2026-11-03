"""Serve the project dashboard without camera capture or simulated telemetry."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import REPO_ROOT, load_config
from .app import AppContext, create_app
from .auth import AuthService
from .hub import StateHub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    default_users = REPO_ROOT / "configs/users.yaml"
    if not default_users.is_file():
        default_users = REPO_ROOT / "configs/users.example.yaml"
    parser.add_argument("--users", type=Path, default=default_users)
    args = parser.parse_args()
    import uvicorn

    ctx = AppContext(cfg=load_config(), hub=StateHub(), auth=AuthService.from_file(args.users))
    uvicorn.run(create_app(ctx), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
