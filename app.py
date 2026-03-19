"""Streamlit 启动入口。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"


def main() -> None:
    """以兼容 `src` 目录布局的方式启动 Streamlit 页面。"""

    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    from labflow.ui.app import run

    run()


if __name__ == "__main__":
    main()
