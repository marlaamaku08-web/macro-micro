"""Macro&Micro тохиргоо.

Бүх тохиргоог environment variable-аас уншина. Утга өгөөгүй үед
development default ашиглана. Нууц үг, credential энд бичихгүй:
SQLite нь файл суурьтай тул credential шаарддаггүй, зөвхөн файлын
зам (DATABASE_PATH) хэрэгтэй.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Model файлууд
MACRO2MICRO_MODEL_PATH = _env_path(
    "MACRO2MICRO_MODEL_PATH", BASE_DIR / "macro2micro_model.pkl"
)
PATHMNIST_MODEL_PATH = _env_path(
    "PATHMNIST_MODEL_PATH", BASE_DIR / "pathmnist_model.keras"
)

# SQLite database
DATABASE_PATH = _env_path("DATABASE_PATH", BASE_DIR / "db" / "macro2micro.db")
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"

# Upload хязгаар
MAX_UPLOAD_MB = _env_float("MAX_UPLOAD_MB", 5.0)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Business logic: үүнээс бага магадлалтай үр дүнг "low_confidence" гэж үзнэ (0–1)
CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.60)

# Model-ийн оролтын зургийн хэмжээ (сургалттай ижил)
IMAGE_SIZE = (64, 64)

# Flask
FLASK_DEBUG = _env_bool("FLASK_DEBUG", True)
PORT = int(os.environ.get("PORT", "5000"))
