"""集中管理项目路径和配置。"""

from __future__ import annotations

from pathlib import Path

# 路径关系：config.py → ui/server/config.py, 父级为 ui/
_UI_DIR = Path(__file__).resolve().parent.parent

PROJECT_ROOT = _UI_DIR.parent
LIVE_DATA_DIR = _UI_DIR / "live_data"
DIST_DIR = _UI_DIR / "dist"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLRUNS_URI = "file:" + str(MLRUNS_DIR)

# CORS 配置（可通过环境变量覆盖）
CORS_ORIGINS: list[str] = ["*"]
