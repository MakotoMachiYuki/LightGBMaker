"""Global configuration & constants for LightGBMaker.

Sensitive keys (such as CARTO_API_KEY) and environment overrides are loaded
strictly on the backend from `LightGBMaker/.env` (falling back to
`website/.env` / OS environment variables / `st.secrets`).

Run with:
    cd LightGBMaker
    streamlit run main.py
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_DIR = BASE_DIR / "Dataset"
LEGACY_DATASET_DIR = ROOT_DIR / "Dataset_2"
WEBSITE_DIR = ROOT_DIR / "website"
WEBSITE_DATASET_DIR = WEBSITE_DIR / "Dataset"
FONT_DIR = BASE_DIR / "Font" if (BASE_DIR / "Font").exists() else WEBSITE_DIR / "Font"
CSS_PATH = BASE_DIR / "style.css"
ENV_PATH = BASE_DIR / ".env"


def resolve_dataset_file(filename: str) -> Path:
    """Resolve a dataset/model file from `LightGBMaker/Dataset` first, falling back to `../Dataset_2`."""
    primary = DATASET_DIR / filename
    if primary.exists():
        return primary
    legacy = LEGACY_DATASET_DIR / filename
    if legacy.exists():
        return legacy
    return primary


def _load_dotenv_file(path: Path) -> None:
    """Load key=value pairs from `.env` using python-dotenv with a native fallback."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=path, override=False)
    except Exception:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv_file(ENV_PATH)
_load_dotenv_file(WEBSITE_DIR / ".env")


def _get_env(key: str, default: str = "") -> str:
    """Read a variable from OS environment / `.env`, falling back to `st.secrets`."""
    val = os.environ.get(key)
    if val is not None and val != "":
        return val
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def _resolve_path(env_key: str, default_path: Path) -> Path:
    raw = _get_env(env_key, "")
    if raw:
        p = Path(raw)
        cand = p if p.is_absolute() else (BASE_DIR / p).resolve()
        if cand.exists():
            return cand
    return default_path


# ── App identity ──────────────────────────────────────────────────────────────
APP_NAME = "LightGBMaker"
APP_TAGLINE = "Klasifikasi Kualitas Jaringan Seluler Indonesia Menggunakan LightGBM"
APP_ICON = None
THESIS_TITLE = (
    "Klasifikasi Kualitas Jaringan Internet Seluler di Indonesia "
    "Menggunakan Algoritma LightGBM"
)

# ── Fonts & Paths ─────────────────────────────────────────────────────────────
FONT_YURUKA_PATH = FONT_DIR / "fot-yuruka-std.ttf"
FONT_NOTO_PATH = FONT_DIR / "NotoSans-VariableFont_wdth,wght.ttf"
FONT_NOTO_ITALIC_PATH = FONT_DIR / "NotoSans-Italic-VariableFont_wdth,wght.ttf"

GDB_PATH = _resolve_path("GDB_PATH", resolve_dataset_file("RBI50K_ADMINISTRASI_KABKOTA_20230907.gdb"))
GDB_LAYER = "ADMINISTRASI_AR_KABKOTA"
GDB_PROVINCE_COL = "WADMPR"

DEFAULT_DATA_PATH = _resolve_path("DATA_PATH", resolve_dataset_file("ookla_indonesia_only.parquet"))
DEFAULT_EXCEL_PATH = _resolve_path("EXCEL_EVAL_PATH", resolve_dataset_file("evaluasi_metrik_lightgbm.xlsx"))

# Quick-load presets from ../Dataset_2
RAW_YEAR_FILES = [
    (2023, "ookla_mobile_2023_Q1_raw.parquet"),
    (2024, "ookla_mobile_2024_Q1_raw.parquet"),
    (2025, "ookla_mobile_2025_Q1_raw.parquet"),
    (2026, "ookla_mobile_2026_Q1_raw.parquet"),
]

QUICK_LOAD_FILES = {
    "Dataset Bersih Khusus Indonesia (2023–2026 · .parquet)": "ookla_indonesia_only.parquet",
    "Dataset Bersih Khusus Indonesia (2023–2026 · .csv)": "ookla_indonesia_only.csv",
    "Seluruh Data Mentah Ookla 2023–2026 Q1 (4 file .parquet)": "__ALL_RAW__",
    "Data Mentah Ookla 2023 Q1 (.parquet)": "ookla_mobile_2023_Q1_raw.parquet",
    "Data Mentah Ookla 2024 Q1 (.parquet)": "ookla_mobile_2024_Q1_raw.parquet",
    "Data Mentah Ookla 2025 Q1 (.parquet)": "ookla_mobile_2025_Q1_raw.parquet",
    "Data Mentah Ookla 2026 Q1 (.parquet)": "ookla_mobile_2026_Q1_raw.parquet",
}

# Default output file names (mirrors Dataset_2/model.ipynb)
EXCEL_EVAL_NAME = "evaluasi_metrik_lightgbm.xlsx"
PDF_REPORT_NAME = "laporan_evaluasi_lightgbm.pdf"
ZIP_NAME = "lightgbmaker_export.zip"
MODEL_PKL_NAMES = {
    "80:20": "lightgbm_ookla_model_80_20.pkl",
    "70:30": "lightgbm_ookla_model_70_30.pkl",
    "main": "lightgbm_ookla_model.pkl",
}
MODEL_BUNDLE_NAMES = {
    "80:20": "lightgbm_ookla_bundle_80_20.pkl",
    "70:30": "lightgbm_ookla_bundle_70_30.pkl",
    "main": "lightgbm_ookla_bundle.pkl",
}
MODEL_TXT_NAMES = {
    "80:20": "lightgbm_ookla_booster_80_20.txt",
    "70:30": "lightgbm_ookla_booster_70_30.txt",
    "main": "lightgbm_ookla_booster.txt",
}

# ── Indonesia bounding box (pre-processing.ipynb) ─────────────────────────────
MIN_LON, MAX_LON = 95.239257, 141.019393
MIN_LAT, MAX_LAT = -10.923083, 5.656182
CRS = "EPSG:4326"
YEARS = [2023, 2024, 2025, 2026]

# ── Geospatial Map (CARTO Voyager + Folium Canvas) ────────────────────────────
MAP_CENTER = [-2.5, 118.0]
MAP_ZOOM = 5
MAP_HEIGHT = 500
MAP_SAMPLE_PER_YEAR = int(_get_env("MAP_SAMPLE_PER_YEAR", "15000"))
TILE_DEG = 360 / 2**16
CARTO_ATTR = "&copy; OpenStreetMap contributors, &copy; CARTO"
_CARTO_API_KEY = _get_env("CARTO_API_KEY", "")
_CARTO_BASEMAP_URL = _get_env(
    "CARTO_BASEMAP_URL",
    "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
)


def get_carto_tiles_url() -> str:
    """Build the CARTO raster tile template URL with CARTO_API_KEY from `.env`."""
    base = _CARTO_BASEMAP_URL
    key = _CARTO_API_KEY
    if not key:
        return base
    if "{key}" in base:
        return base.replace("{key}", key)
    parts = urlsplit(base)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.setdefault("key", key)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


# ── Features, target & labels (model.ipynb) ───────────────────────────────────
FEATURES = [
    "avg_d_kbps",
    "avg_u_kbps",
    "avg_lat_ms",
    "avg_lat_down_ms",
    "avg_lat_up_ms",
    "tests",
    "devices",
]
TARGET = "quality"
TARGET_ENCODED = "quality_encoded"
REQUIRED_CLEAN_COLS = FEATURES + [TARGET]
DROP_COLS = ["quadkey", "tile", "geometry", "index_right"]

LABEL_MAPPING = {"Bad": 0, "Serviceable": 1, "Average": 2, "Good": 3, "Great": 4}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
CLASS_NAMES = [INVERSE_LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]
CLASS_LABELS = [f"{i} - {c}" for i, c in enumerate(CLASS_NAMES)]
MINORITY_CLASSES = ["Good", "Great"]

FEATURE_INFO = {
    "avg_d_kbps": ("Download (kbps)", "Rata-rata kecepatan unduh"),
    "avg_u_kbps": ("Upload (kbps)", "Rata-rata kecepatan unggah"),
    "avg_lat_ms": ("Latency (ms)", "Idle latency"),
    "avg_lat_down_ms": ("DL Latency (ms)", "Loaded download latency"),
    "avg_lat_up_ms": ("UL Latency (ms)", "Loaded upload latency"),
    "tests": ("Tests", "Jumlah pengujian per tile"),
    "devices": ("Devices", "Jumlah perangkat per tile"),
}

# ── Split schemes, CV & default hyper-parameters ──────────────────────────────
SPLIT_RATIOS = {"80:20": 0.20, "70:30": 0.30}
SPLIT_NAMES = list(SPLIT_RATIOS.keys())
RANDOM_STATE = int(_get_env("RANDOM_STATE", "727"))
K_FOLDS = int(_get_env("K_FOLDS", "5"))
EARLY_STOPPING = int(_get_env("EARLY_STOPPING", "50"))

DEFAULT_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 5,
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

PARAM_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [-1, 6, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500, 800],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0],
}

# ── Palettes (website/style.css + website/config.py) ─────────────────────────
QUALITY_COLORS = {
    "Bad": "#212121",
    "Serviceable": "#e53935",
    "Average": "#fb8c00",
    "Good": "#1e88e5",
    "Great": "#43a047",
}
SPLIT_COLORS = {"80:20": "#4da8fa", "70:30": "#7c3aed"}
INK = "#0b2545"
MUTED = "#48607e"
TBL = {
    "bg": "#181e38",
    "bg_alt": "#1f2747",
    "head": "#28255a",
    "text": "#eef2ff",
    "border": "#333c6b",
    "hl_col": "#3b1d6e",
    "you_bg": "#581c87",
    "outer": "#5b5bd6",
}

QUALITY_INTERPRETATION = {
    "Bad": (
        "Degradasi transmisi sangat berat — hanya mendukung transmisi data ringan "
        "(teks/notifikasi); belum mampu memfasilitasi multimedia interaktif."
    ),
    "Serviceable": (
        "Fungsional minimum untuk browsing, media sosial dan pesan teks, namun "
        "terkendala pada streaming multimedia berkelanjutan."
    ),
    "Average": (
        "Konektivitas standar untuk produktivitas harian, e-learning dan video "
        "daring tanpa gangguan signifikan."
    ),
    "Good": (
        "Kapasitas tinggi dan stabil untuk video streaming resolusi tinggi "
        "(YouTube/Twitch) tanpa jeda pemuatan."
    ),
    "Great": (
        "Konektivitas paling optimal dengan delay & jitter terendah — ideal untuk "
        "komunikasi real-time, video kualitas maksimal dan unduhan besar."
    ),
}

# Preset traffic profiles for the Classification tab (one per class)
PRESETS = {
    "Bad": {"avg_d_kbps": 7571, "avg_u_kbps": 251, "avg_lat_ms": 209,
            "avg_lat_down_ms": 305.0, "avg_lat_up_ms": 259.0, "tests": 2, "devices": 1},
    "Serviceable": {"avg_d_kbps": 3545, "avg_u_kbps": 3785, "avg_lat_ms": 133,
                    "avg_lat_down_ms": 804.0, "avg_lat_up_ms": 507.0, "tests": 9, "devices": 2},
    "Average": {"avg_d_kbps": 82080, "avg_u_kbps": 50798, "avg_lat_ms": 20,
                "avg_lat_down_ms": 290.0, "avg_lat_up_ms": 111.0, "tests": 9, "devices": 1},
    "Good": {"avg_d_kbps": 148748, "avg_u_kbps": 33726, "avg_lat_ms": 31,
             "avg_lat_down_ms": 210.0, "avg_lat_up_ms": 847.0, "tests": 1, "devices": 1},
    "Great": {"avg_d_kbps": 218083, "avg_u_kbps": 56552, "avg_lat_ms": 5,
              "avg_lat_down_ms": 167.0, "avg_lat_up_ms": 527.0, "tests": 2, "devices": 1},
}
