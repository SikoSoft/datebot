from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root for all personal data (db, photos, models). Override with DATEBOT_HOME."""
    root = Path(os.environ.get("DATEBOT_HOME", Path.cwd() / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_dir() / "datebot.db"


def photos_dir() -> Path:
    path = data_dir() / "photos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
