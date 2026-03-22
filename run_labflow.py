"""LabFlow 通用启动入口。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "8501"


def ensure_env_file() -> None:
    """首次运行时自动生成 `.env` 模板，减少手工准备步骤。"""

    if ENV_FILE.exists() or not ENV_EXAMPLE_FILE.exists():
        return
    shutil.copyfile(ENV_EXAMPLE_FILE, ENV_FILE)


def main() -> int:
    ensure_env_file()
    host = os.environ.get("LABFLOW_HOST", DEFAULT_HOST)
    port = os.environ.get("LABFLOW_PORT", DEFAULT_PORT)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        host,
        "--server.port",
        port,
        "--server.headless",
        "true",
    ]
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except ModuleNotFoundError:
        print("当前环境缺少 Streamlit，请先执行：pip install -r requirements.txt")
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
