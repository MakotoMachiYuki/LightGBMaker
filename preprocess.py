"""Pre-processing pipeline (pre-processing.ipynb + indonesia.ipynb).

Stages
------
1. Multi-format loading (.parquet / .xlsx / .xls / .csv) + `year` tagging.
2. Indonesia bounding-box filter (tile_x / tile_y or WKT `tile`, EPSG:4326).
3. Exact administrative-polygon spatial join against the RBI50K Kab/Kota GDB
   (`gpd.sjoin(..., how="inner", predicate="intersects")`, WADMPR -> Provinsi).
4. Cleaning (drop heavy columns, NaN report + drop, remove 'Unknown').
5. Ordinal target encoding (Bad=0 … Great=4).
6. Summary tables (class distribution, province × quality, CSV vs Parquet).
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg

ProgressFn = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:  # pragma: no cover - trivial
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════
def detect_year(name: str) -> int | None:
    m = re.search(r"(20[12]\d)", str(name))
    return int(m.group(1)) if m else None


def _read_parquet(src, bbox_pushdown: bool) -> pd.DataFrame:
    """Read a parquet file; optionally push the Indonesia bbox down to pyarrow.

    Raw Ookla quarterly files hold ~3.7 M global rows (≈200 MB). Pushing the
    bounding box (and skipping the heavy `quadkey` column) into the reader keeps
    memory low; the result is identical to `gdf.cx[...]` on the tile centroid.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    names = pf.schema_arrow.names
    has_xy = {"tile_x", "tile_y"}.issubset(names)
    already_verified = "Provinsi" in names and "tile" not in names
    cols = [c for c in names if c != "quadkey"]
    if bbox_pushdown and has_xy and not already_verified:
        cols = [c for c in cols if c != "tile"]
        pad = 360.0 / (2**16)  # zoom-16 tile width (~0.0055°) so edge polygons match gdf.cx
        filters = [
            ("tile_x", ">=", cfg.MIN_LON - pad), ("tile_x", "<=", cfg.MAX_LON + pad),
            ("tile_y", ">=", cfg.MIN_LAT - pad), ("tile_y", "<=", cfg.MAX_LAT + pad),
        ]
        table = pq.read_table(src, columns=cols, filters=filters)
        df = table.to_pandas()
        df.attrs["rows_before_bbox"] = pf.metadata.num_rows
        return df
    return pq.read_table(src, columns=cols).to_pandas()


def read_any(src, name: str, bbox_pushdown: bool = True) -> pd.DataFrame:
    """Load a CSV / Excel / Parquet source (path or uploaded buffer)."""
    ext = Path(name).suffix.lower()
    if ext == ".parquet":
        return _read_parquet(src, bbox_pushdown)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(src)
    if ext == ".csv":
        return pd.read_csv(src, low_memory=False)
    raise ValueError(f"Unsupported file type: {name}")


@st.cache_data(show_spinner=False, max_entries=8)
def load_workspace_file(file_name: str, bbox_pushdown: bool = True) -> pd.DataFrame:
    path = cfg.resolve_dataset_file(file_name)
    return read_any(path, file_name, bbox_pushdown)


def load_uploaded(upload, bbox_pushdown: bool = True) -> pd.DataFrame:
    data = io.BytesIO(upload.getvalue())
    return read_any(data, upload.name, bbox_pushdown)


# ══════════════════════════════════════════════════════════════════════════════
# Spatial helpers
# ══════════════════════════════════════════════════════════════════════════════
def geo_available() -> bool:
    try:
        import geopandas  # noqa: F401
        import pyogrio  # noqa: F401

        return True
    except Exception:
        return False


def ensure_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee `tile_x` / `tile_y` (lon / lat) columns, deriving from WKT if needed."""
    if {"tile_x", "tile_y"}.issubset(df.columns):
        return df
    if "tile" in df.columns and geo_available():
        import geopandas as gpd

        geom = gpd.GeoSeries.from_wkt(df["tile"], crs=cfg.CRS)
        cent = geom.representative_point()
        df = df.copy()
        df["tile_x"] = cent.x.round(4)
        df["tile_y"] = cent.y.round(4)
        return df
    return df


def bbox_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "tile" in df.columns and geo_available():
        import geopandas as gpd

        geom = gpd.GeoSeries.from_wkt(df["tile"], crs=cfg.CRS)
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs=cfg.CRS)
        return pd.DataFrame(gdf.cx[cfg.MIN_LON:cfg.MAX_LON, cfg.MIN_LAT:cfg.MAX_LAT].drop(columns=["geometry"]))
    if not {"tile_x", "tile_y"}.issubset(df.columns):
        return df
    pad = 360.0 / (2**16)
    m = (
        df["tile_x"].between(cfg.MIN_LON - pad, cfg.MAX_LON + pad)
        & df["tile_y"].between(cfg.MIN_LAT - pad, cfg.MAX_LAT + pad)
    )
    return df.loc[m]


@st.cache_resource(show_spinner=False)
def load_admin_boundaries():
    """Load RBI50K Kab/Kota polygons (WADMPR only), exploded for a tight STRtree."""
    import geopandas as gpd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(
            cfg.GDB_PATH, layer=cfg.GDB_LAYER, engine="pyogrio",
            columns=[cfg.GDB_PROVINCE_COL],
        )
    # Compound CRS "WGS 84 + EGM2008 height" -> horizontal component is EPSG:4326
    try:
        horiz = gdf.crs.sub_crs_list[0] if gdf.crs.is_compound else gdf.crs
        if horiz.to_epsg() == 4326:
            gdf = gdf.set_crs(cfg.CRS, allow_override=True)
        else:
            gdf = gdf.to_crs(cfg.CRS)
    except Exception:
        gdf = gdf.set_crs(cfg.CRS, allow_override=True)
    gdf["geometry"] = gdf.geometry.force_2d()
    gdf = gdf[gdf[cfg.GDB_PROVINCE_COL].fillna("").str.strip() != ""]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf[[cfg.GDB_PROVINCE_COL, "geometry"]]


def spatial_join_indonesia(df: pd.DataFrame, progress: ProgressFn = _noop) -> pd.DataFrame:
    """Keep only points intersecting an Indonesian Kab/Kota polygon; add `Provinsi`.

    Semantically identical to indonesia.ipynb's
    `gpd.sjoin(points, kabkota, how="inner", predicate="intersects")`, but ~100×
    faster: it joins the *unique* tile coordinates only (tiles repeat across
    years), builds the STRtree on the points and queries it with the *prepared*
    (exploded) polygons. Points lying exactly on a shared border keep their
    first match, so no rows are duplicated.
    """
    import shapely

    progress(0.0, "Memuat poligon administrasi RBI50K (GDB)…")
    admin = load_admin_boundaries()
    uniq = df[["tile_x", "tile_y"]].drop_duplicates().reset_index(drop=True)
    progress(0.35, f"Spatial join (intersects) pada {len(uniq):,} titik koordinat unik…")
    pts = shapely.points(uniq["tile_x"].to_numpy(), uniq["tile_y"].to_numpy())
    tree = shapely.STRtree(pts)
    polys = admin.geometry.values
    shapely.prepare(polys)
    poly_idx, pt_idx = tree.query(polys, predicate="intersects")
    order = np.lexsort((poly_idx, pt_idx))
    pt_s, poly_s = pt_idx[order], poly_idx[order]
    first = np.r_[True, pt_s[1:] != pt_s[:-1]] if len(pt_s) else np.array([], dtype=bool)
    lookup = uniq.iloc[pt_s[first]].copy()
    lookup[cfg.GDB_PROVINCE_COL] = admin[cfg.GDB_PROVINCE_COL].to_numpy()[poly_s[first]]
    progress(0.9, "Memetakan WADMPR ke Provinsi…")
    out = df.drop(columns=["Provinsi"], errors="ignore").merge(
        lookup, on=["tile_x", "tile_y"], how="inner"
    )
    out = out.rename(columns={cfg.GDB_PROVINCE_COL: "Provinsi"})
    progress(1.0, "Spatial join selesai.")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SourceFrame:
    name: str
    df: pd.DataFrame
    year: int | None = None


@dataclass
class PipelineOptions:
    do_bbox: bool = True
    do_sjoin: bool = True
    reverify_provinsi: bool = True
    run_benchmark: bool = True


@dataclass
class PipelineResult:
    df: pd.DataFrame
    stats: dict
    missing_report: pd.DataFrame
    dist_df: pd.DataFrame
    prov_quality_df: pd.DataFrame
    storage_df: pd.DataFrame | None
    log: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    return pd.DataFrame({
        "Label Numerik": [cfg.LABEL_MAPPING[k] for k in cfg.CLASS_NAMES],
        "Kelas Kualitas": cfg.CLASS_NAMES,
        "Jumlah Sampel": [int((df[cfg.TARGET] == k).sum()) for k in cfg.CLASS_NAMES],
        "Persentase (%)": [round((df[cfg.TARGET] == k).sum() / n * 100, 2) for k in cfg.CLASS_NAMES],
    })


def province_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "Provinsi" not in df.columns:
        return pd.DataFrame()
    tab = pd.crosstab(df["Provinsi"], df[cfg.TARGET])
    tab = tab.reindex(columns=cfg.CLASS_NAMES, fill_value=0)
    tab["Total"] = tab.sum(axis=1)
    tab = tab.sort_values("Total", ascending=False)
    tab.index.name = "Provinsi"
    return tab.reset_index()


def storage_benchmark(df: pd.DataFrame, base_name: str = "ookla_indonesia_only") -> pd.DataFrame:
    """CSV vs Parquet: file size, read & write time (model.ipynb TAHAP 1)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_p = os.path.join(tmp, f"{base_name}.csv")
        pq_p = os.path.join(tmp, f"{base_name}.parquet")
        t = time.perf_counter(); df.to_csv(csv_p, index=False); csv_w = time.perf_counter() - t
        t = time.perf_counter(); pd.read_csv(csv_p, low_memory=False); csv_r = time.perf_counter() - t
        t = time.perf_counter(); df.to_parquet(pq_p, index=False); pq_w = time.perf_counter() - t
        t = time.perf_counter(); pd.read_parquet(pq_p); pq_r = time.perf_counter() - t
        csv_mb = os.path.getsize(csv_p) / (1024 * 1024)
        pq_mb = os.path.getsize(pq_p) / (1024 * 1024)
    saving = (1 - pq_mb / csv_mb) * 100 if csv_mb else 0.0
    return pd.DataFrame([
        {"Format File": "CSV (.csv)", "Nama File": f"{base_name}.csv",
         "Ukuran File (MB)": round(csv_mb, 2), "Waktu Baca (detik)": round(csv_r, 4),
         "Waktu Tulis (detik)": round(csv_w, 4), "Penghematan Ruang (%)": 0.0},
        {"Format File": "Parquet (.parquet)", "Nama File": f"{base_name}.parquet",
         "Ukuran File (MB)": round(pq_mb, 2), "Waktu Baca (detik)": round(pq_r, 4),
         "Waktu Tulis (detik)": round(pq_w, 4), "Penghematan Ruang (%)": round(saving, 2)},
    ])


def run_pipeline(
    sources: list[SourceFrame],
    opts: PipelineOptions,
    progress: ProgressFn = _noop,
) -> PipelineResult:
    log: list[str] = []
    stats: dict = {}

    # ── 1. Concatenate with year tagging ──────────────────────────────────────
    progress(0.02, "Menggabungkan data & menandai tahun…")
    frames, raw_total = [], 0
    for s in sources:
        d = s.df
        raw_total += int(d.attrs.get("rows_before_bbox", len(d)))
        if s.year is not None:
            d = d.copy()
            d["year"] = int(s.year)
        frames.append(d)
        log.append(f"Memuat {s.name}: {len(d):,} baris" + (f" (year={s.year})" if s.year else ""))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    pushed = raw_total - len(df)
    stats["initial_rows"] = raw_total
    if pushed:
        log.append(f"Filter Bounding Box awal menghapus {pushed:,} baris di luar Indonesia.")

    if cfg.TARGET not in df.columns:
        raise ValueError("Kolom 'quality' tidak ditemukan pada dataset.")

    # ── 2. Bounding box ───────────────────────────────────────────────────────
    has_prov = "Provinsi" in df.columns
    skip_spatial = has_prov and not opts.reverify_provinsi
    progress(0.08, "Memeriksa koordinat spasial (tile_x / tile_y)…")
    df = ensure_xy(df)
    has_xy = {"tile_x", "tile_y"}.issubset(df.columns)
    bbox_removed = pushed
    if opts.do_bbox and has_xy and not skip_spatial:
        progress(0.12, "Tahap 1 · Filter Bounding Box Indonesia…")
        before = len(df)
        df = bbox_filter(df)
        bbox_removed += before - len(df)
        log.append(f"Bounding box [{cfg.MIN_LON}, {cfg.MAX_LON}] × [{cfg.MIN_LAT}, {cfg.MAX_LAT}] "
                   f"menghapus {before - len(df):,} baris.")
    elif not has_xy:
        log.append("Kolom tile_x/tile_y atau WKT 'tile' tidak ditemukan — filter spasial dilewati.")
    stats["bbox_removed"] = int(bbox_removed)

    # ── 3. Exact admin polygon join ───────────────────────────────────────────
    sjoin_removed = 0
    if opts.do_sjoin and has_xy and not skip_spatial:
        if not geo_available():
            log.append("geopandas/pyogrio belum terpasang — spatial join poligon dilewati.")
        elif not cfg.GDB_PATH.exists():
            log.append(f"File GDB tidak ditemukan di {cfg.GDB_PATH} — spatial join poligon dilewati.")
        else:
            before = len(df)
            df = spatial_join_indonesia(
                df, lambda p, m: progress(0.15 + 0.5 * p, f"Tahap 2 · {m}")
            )
            sjoin_removed = before - len(df)
            log.append(f"Spatial join poligon (RBI50K {cfg.GDB_LAYER}) menghapus "
                       f"{sjoin_removed:,} baris di laut lepas / negara tetangga.")
    elif skip_spatial:
        log.append("Kolom 'Provinsi' yang sudah terverifikasi dipertahankan.")
    stats["sjoin_removed"] = int(sjoin_removed)
    stats["non_indonesia_removed"] = int(bbox_removed + sjoin_removed)

    # ── 4. Cleaning ───────────────────────────────────────────────────────────
    progress(0.7, "Membersihkan kolom redundan & missing values…")
    df = df.drop(columns=[c for c in cfg.DROP_COLS if c in df.columns])
    missing_report = pd.DataFrame({
        "Kolom": df.columns,
        "Missing (NaN)": [int(df[c].isna().sum()) for c in df.columns],
        "Persentase (%)": [round(df[c].isna().mean() * 100, 4) if len(df) else 0.0 for c in df.columns],
        "Tipe Data": [str(df[c].dtype) for c in df.columns],
    })
    subset = [c for c in cfg.REQUIRED_CLEAN_COLS if c in df.columns]
    missing_cols = [c for c in cfg.REQUIRED_CLEAN_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    before = len(df)
    df = df.dropna(subset=subset)
    nan_removed = before - len(df)
    before = len(df)
    df[cfg.TARGET] = df[cfg.TARGET].astype(str).str.strip()
    df = df[df[cfg.TARGET] != "Unknown"]
    df = df[df[cfg.TARGET].isin(cfg.LABEL_MAPPING.keys())]
    unknown_removed = before - len(df)
    stats["nan_removed"] = int(nan_removed)
    stats["unknown_removed"] = int(unknown_removed)
    stats["missing_removed"] = int(nan_removed + unknown_removed)
    log.append(f"Dropped {nan_removed:,} NaN rows and {unknown_removed:,} 'Unknown'/invalid-label rows.")

    # ── 5. Ordinal encoding ───────────────────────────────────────────────────
    progress(0.8, "Encoding quality → quality_encoded (Bad=0 … Great=4)…")
    df[cfg.TARGET_ENCODED] = df[cfg.TARGET].map(cfg.LABEL_MAPPING).astype("int64")
    for c in ("tests", "devices"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.reset_index(drop=True)

    stats["final_rows"] = int(len(df))
    stats["provinces"] = int(df["Provinsi"].nunique()) if "Provinsi" in df.columns else 0
    stats["years"] = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df.columns else []

    # ── 6. Summaries ──────────────────────────────────────────────────────────
    progress(0.86, "Building distribution tables…")
    dist_df = class_distribution(df)
    prov_df = province_quality(df)
    storage_df = None
    if opts.run_benchmark and len(df):
        progress(0.9, "Benchmarking CSV vs Parquet storage & I/O…")
        storage_df = storage_benchmark(df)
    progress(1.0, "Pre-processing complete.")
    return PipelineResult(
        df=df, stats=stats, missing_report=missing_report, dist_df=dist_df,
        prov_quality_df=prov_df, storage_df=storage_df, log=log,
        sources=[s.name for s in sources],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset export helpers
# ══════════════════════════════════════════════════════════════════════════════
EXCEL_MAX_ROWS = 1_048_575


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet: str = "data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        if len(df) <= EXCEL_MAX_ROWS:
            df.to_excel(xw, sheet_name=sheet, index=False)
        else:  # split across sheets (Excel row limit)
            for i, start in enumerate(range(0, len(df), EXCEL_MAX_ROWS), start=1):
                df.iloc[start:start + EXCEL_MAX_ROWS].to_excel(
                    xw, sheet_name=f"{sheet}_{i}", index=False
                )
    return buf.getvalue()
