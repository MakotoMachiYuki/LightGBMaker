"""Exports: multi-sheet Excel, multi-page PDF report, model files & ZIP archive."""

from __future__ import annotations

import io
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

import config as cfg
import modelling


# ══════════════════════════════════════════════════════════════════════════════
# Excel
# ══════════════════════════════════════════════════════════════════════════════
def _placeholder(msg: str) -> pd.DataFrame:
    return pd.DataFrame({"Keterangan": [msg]})


def recommendation_frame(results: dict, rec: dict, primary: str) -> pd.DataFrame:
    crit = modelling.criteria_frame(rec)
    head = pd.DataFrame([
        {"Metrik Evaluasi": "MODEL REKOMENDASI", "Kategori": "-", "Bobot": None,
         "Model Unggul": rec["winner"]},
        {"Metrik Evaluasi": "MODEL UTAMA (EKSPOR)", "Kategori": "-", "Bobot": None,
         "Model Unggul": primary},
        *[{"Metrik Evaluasi": f"Skor Total {k}", "Kategori": "-", "Bobot": None,
           "Model Unggul": round(v, 3)} for k, v in rec["scores"].items()],
    ])
    return pd.concat([head, crit], ignore_index=True)


def excel_bytes(results: dict, rec: dict, primary: str, pp=None) -> bytes:
    sp = results["splits"]
    buf = io.BytesIO()

    def get(split: str, key: str, fallback_msg: str) -> pd.DataFrame:
        v = sp.get(split, {}).get(key)
        return v if isinstance(v, pd.DataFrame) else _placeholder(fallback_msg)

    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        results["comparison_df"].to_excel(xw, sheet_name="Perbandingan_Global", index=False)
        cvo = results.get("cv_overview_df")
        (cvo if cvo is not None and not cvo.empty else _placeholder("K-Fold CV tidak dijalankan.")
         ).to_excel(xw, sheet_name="KFold_Ringkasan", index=False)
        get("80:20", "cv_df", "K-Fold 80:20 tidak tersedia.").to_excel(xw, sheet_name="KFold_80_20", index=False)
        get("70:30", "cv_df", "K-Fold 70:30 tidak tersedia.").to_excel(xw, sheet_name="KFold_70_30", index=False)
        cvc = results.get("cv_combined_df")
        (cvc if cvc is not None and not cvc.empty else _placeholder("K-Fold CV tidak dijalankan.")
         ).to_excel(xw, sheet_name="KFold_Gabungan", index=False)
        get("80:20", "ovr_df", "-").to_excel(xw, sheet_name="OVR_Kelas_80_20", index=False)
        get("70:30", "ovr_df", "-").to_excel(xw, sheet_name="OVR_Kelas_70_30", index=False)
        get("80:20", "cm_df", "-").to_excel(xw, sheet_name="Confusion_Matrix_80_20")
        get("70:30", "cm_df", "-").to_excel(xw, sheet_name="Confusion_Matrix_70_30")
        results["fi_df"].to_excel(xw, sheet_name="Feature_Importance", index=False)
        results["dist_df"].to_excel(xw, sheet_name="Distribusi_Kelas", index=False)
        results["split_summary_df"].to_excel(xw, sheet_name="Distribusi_Split", index=False)
        st_df = getattr(pp, "storage_df", None) if pp is not None else None
        (st_df if st_df is not None else _placeholder(
            "Benchmark CSV vs Parquet belum dijalankan pada tahap pre-processing.")
         ).to_excel(xw, sheet_name="CSV_vs_Parquet", index=False)
        recommendation_frame(results, rec, primary).to_excel(
            xw, sheet_name="Rekomendasi_Model", index=False)
        # Auto-fit column widths
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) if c.value is not None else 0) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Model files
# ══════════════════════════════════════════════════════════════════════════════
def build_bundle(results: dict, split: str) -> dict:
    r = results["splits"][split]
    return {
        "model": r["model"],
        "selected_split": split,
        "features": list(cfg.FEATURES),
        "target": cfg.TARGET,
        "label_mapping": dict(cfg.LABEL_MAPPING),
        "inverse_label_mapping": dict(cfg.INVERSE_LABEL_MAPPING),
        "class_names": list(cfg.CLASS_NAMES),
        "hyperparameters": dict(r["params"]),
        "metrics": {
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "weighted_f1": float(r["weighted_f1"]),
            "train_time_sec": float(r["train_time_sec"]),
            "pred_time_sec": float(r["pred_time_sec"]),
        },
    }


def _dump(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def model_files(results: dict, primary: str) -> dict[str, bytes]:
    """Return {file_name: bytes} for every model artefact (per split + main)."""
    out: dict[str, bytes] = {}
    for split in results["splits"]:
        mdl = results["splits"][split]["model"]
        out[cfg.MODEL_PKL_NAMES[split]] = _dump(mdl)
        out[cfg.MODEL_BUNDLE_NAMES[split]] = _dump(build_bundle(results, split))
        out[cfg.MODEL_TXT_NAMES[split]] = mdl.booster_.model_to_string().encode("utf-8")
    mdl = results["splits"][primary]["model"]
    out[cfg.MODEL_PKL_NAMES["main"]] = _dump(mdl)
    out[cfg.MODEL_BUNDLE_NAMES["main"]] = _dump(build_bundle(results, primary))
    out[cfg.MODEL_TXT_NAMES["main"]] = mdl.booster_.model_to_string().encode("utf-8")
    return out


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PDF report (ReportLab)
# ══════════════════════════════════════════════════════════════════════════════
def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401

        return True
    except Exception:
        return False


def _register_pdf_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    title, body = "Helvetica-Bold", "Helvetica"
    try:
        if cfg.FONT_YURUKA_PATH.exists():
            if "Yuruka" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Yuruka", str(cfg.FONT_YURUKA_PATH)))
            title = "Yuruka"
    except Exception:
        pass
    return title, body


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.2f}" if abs(float(v)) >= 1000 else f"{float(v):.4f}"
    return str(v)


def pdf_bytes(results: dict, rec: dict, primary: str, charts_png: dict[str, bytes],
              pp=None) -> bytes:
    import re

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    title_font, body_font = _register_pdf_fonts()
    page = landscape(A4)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=page, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Laporan Evaluasi LightGBM", author=cfg.APP_NAME)
    avail_w = page[0] - 2.8 * cm
    ss = getSampleStyleSheet()
    ink = colors.HexColor(cfg.INK)
    H0 = ParagraphStyle("H0", parent=ss["Title"], fontName=title_font, fontSize=24, textColor=ink, leading=30)
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName=title_font, fontSize=16, textColor=ink, spaceBefore=6, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=title_font, fontSize=12.5, textColor=ink, spaceBefore=6, spaceAfter=4)
    P = ParagraphStyle("P", parent=ss["BodyText"], fontName=body_font, fontSize=9.5, leading=13.5, textColor=ink)
    SMALL = ParagraphStyle("S", parent=P, fontSize=8, textColor=colors.HexColor(cfg.MUTED))
    CELL = ParagraphStyle("C", parent=P, fontSize=7.4, leading=9, alignment=TA_CENTER, textColor=colors.HexColor(cfg.TBL["text"]))
    CELLH = ParagraphStyle("CH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)

    def clean(text: str) -> str:
        text = re.sub(r"</?(span|div)[^>]*>", "", text)
        return text.replace("σ", "std").replace("&amp;", "&amp;")

    def df_table(df: pd.DataFrame, index: bool = False, max_cols: int | None = None) -> Table:
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [" · ".join(map(str, c)) for c in d.columns]
        if index:
            d = d.reset_index()
        if max_cols:
            d = d.iloc[:, :max_cols]
        data = [[Paragraph(str(c), CELLH) for c in d.columns]]
        for _, r in d.iterrows():
            data.append([Paragraph(_fmt(v), CELL) for v in r.tolist()])
        n = len(d.columns)
        t = Table(data, colWidths=[avail_w / n] * n, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(cfg.TBL["head"])),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(cfg.TBL["bg"]), colors.HexColor(cfg.TBL["bg_alt"])]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(cfg.TBL["border"])),
            ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(cfg.TBL["outer"])),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    def img(key: str, width: float = avail_w, max_h: float = 15 * cm):
        data = charts_png.get(key)
        if not data:
            return Spacer(1, 1)
        from reportlab.lib.utils import ImageReader

        ir = ImageReader(io.BytesIO(data))
        iw, ih = ir.getSize()
        w = width
        h = w * ih / iw
        if h > max_h:
            h = max_h
            w = h * iw / ih
        return Image(io.BytesIO(data), width=w, height=h)

    sp = results["splits"]
    story = []
    # ── Cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("Laporan Evaluasi Model LightGBM", H0),
        Paragraph(cfg.THESIS_TITLE, H2),
        Paragraph(f"Dibuat oleh {cfg.APP_NAME} · {time.strftime('%d %B %Y, %H:%M:%S')}", SMALL),
        Spacer(1, 0.6 * cm),
    ]
    # Dataset & pre-processing summary
    story.append(Paragraph("1. Ringkasan Dataset & Pre-processing", H1))
    cfgd = results["config"]
    rows = [["Jumlah baris data latih+uji", f"{cfgd['n_rows']:,}"],
            ["Fitur prediktor", ", ".join(cfg.FEATURES)],
            ["Target", "quality → quality_encoded (Bad=0, Serviceable=1, Average=2, Good=3, Great=4)"],
            ["Sampling fraksi data", f"{cfgd['sample_frac'] * 100:.0f}%"]]
    if pp is not None:
        s = pp.stats
        rows = [["Baris awal (sumber)", f"{s['initial_rows']:,}"],
                ["Dihapus: luar Indonesia / laut (bbox + poligon)", f"{s['non_indonesia_removed']:,}"],
                ["Dihapus: missing value / Unknown", f"{s['missing_removed']:,}"],
                ["Baris bersih Indonesia", f"{s['final_rows']:,}"],
                ["Jumlah provinsi", f"{s['provinces']}"]] + rows
    t = Table([[Paragraph(a, P), Paragraph(b, P)] for a, b in rows], colWidths=[avail_w * 0.35, avail_w * 0.65])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fb3c8")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e6f3ff"))]))
    story += [t, Spacer(1, 0.3 * cm)]
    story += [Paragraph("Distribusi Kelas", H2), df_table(results["dist_df"])]
    if pp is not None and pp.storage_df is not None:
        story += [Paragraph("Benchmark CSV vs Parquet", H2), df_table(pp.storage_df)]

    # Hyper-parameters
    params = {k: v for k, v in cfgd["params"].items() if k not in ("verbose", "n_jobs")}
    story += [Paragraph("Konfigurasi Hyperparameter", H2),
              df_table(pd.DataFrame([{k: str(v) for k, v in params.items()}])),
              Paragraph(f"Early stopping: {cfgd['early_stopping']} · K-Fold: {cfgd['k_folds']} "
                        f"({'aktif' if cfgd['run_cv'] else 'nonaktif'}) · Tuning: "
                        f"{cfgd['search_method'] if cfgd['tuning'] else 'nonaktif'} · "
                        f"random_state={cfg.RANDOM_STATE}", SMALL)]
    story.append(PageBreak())

    # ── Recommendation ────────────────────────────────────────────────────────
    story.append(Paragraph("2. Rekomendasi Model Otomatis", H1))
    story.append(Paragraph(
        f"<b>Model Terbaik: {rec['winner']}</b> · Model Utama Ekspor: "
        f"<b>{primary}</b> · Skor Evaluasi: "
        + ", ".join(f"{k} = {v:.2f}" for k, v in rec["scores"].items()),
        P,
    ))
    crit = modelling.criteria_frame(rec)
    if not crit.empty:
        story += [Spacer(1, 0.25 * cm), df_table(crit)]
    story.append(PageBreak())

    # ── Global comparison ─────────────────────────────────────────────────────
    story.append(Paragraph("3. Perbandingan Performa Global (80:20 vs 70:30)", H1))
    story.append(df_table(results["comparison_df"]))
    story += [Spacer(1, 0.3 * cm), img("global", max_h=8 * cm), img("time", max_h=6 * cm)]
    story.append(PageBreak())
    story += [Paragraph("Kurva Konvergensi Multi-Logloss", H2), img("logloss", max_h=8.5 * cm),
              Paragraph("Distribusi Data Latih & Uji", H2), df_table(results["split_pivot_df"], index=True)]
    story.append(PageBreak())

    # ── Per-class OVR + Confusion matrix ─────────────────────────────────────
    story.append(Paragraph("4. Evaluasi per Kelas (One-vs-Rest) & Confusion Matrix", H1))
    for s, r in sp.items():
        story += [Paragraph(f"Tabel OVR Skema {s}", H2), df_table(r["ovr_df"])]
    story += [Spacer(1, 0.2 * cm), img("per_class", max_h=7.5 * cm)]
    story.append(PageBreak())
    story += [Paragraph("Confusion Matrix 5×5", H2), img("cm", max_h=11 * cm)]
    for s, r in sp.items():
        story += [KeepTogether([Paragraph(f"Confusion Matrix Skema {s}", H2), df_table(r["cm_df"], index=True)])]
    story.append(PageBreak())

    # ── K-Fold ────────────────────────────────────────────────────────────────
    cvo = results.get("cv_overview_df")
    if cvo is not None and not cvo.empty:
        story.append(Paragraph(f"5. K-Fold Cross Validation (k = {cfgd['k_folds']})", H1))
        story.append(df_table(cvo))
        for s, r in sp.items():
            if r.get("cv_df") is not None:
                story += [Paragraph(f"Detail K-Fold Skema {s}", H2), df_table(r["cv_df"])]
        story.append(PageBreak())
        story += [img("kfold_line", max_h=7 * cm), Spacer(1, 0.3 * cm), img("kfold_bar", max_h=8 * cm)]
        story.append(PageBreak())

    # ── Feature importance ────────────────────────────────────────────────────
    story.append(Paragraph("6. Feature Importance (Split & Gain)", H1))
    story += [df_table(results["fi_df"]), Spacer(1, 0.3 * cm), img("fi", max_h=11 * cm)]

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(cfg.MUTED))
        canvas.drawString(1.4 * cm, 0.7 * cm, f"{cfg.APP_NAME} · Laporan Evaluasi LightGBM")
        canvas.drawRightString(page[0] - 1.4 * cm, 0.7 * cm, f"Halaman {doc_.page}")
        canvas.setStrokeColor(colors.HexColor("#9ae6ef"))
        canvas.setLineWidth(3)
        canvas.line(1.4 * cm, page[1] - 0.8 * cm, page[0] - 1.4 * cm, page[1] - 0.8 * cm)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Direct 1-click export panel (used in the "Export" tab)
# ══════════════════════════════════════════════════════════════════════════════
def render_export_panel(pp, res: dict, rec: dict, ch: dict, key_prefix: str = "exp") -> None:
    import streamlit as st
    import charts
    import preprocess as pp_mod
    import ui

    primary = rec["winner"]
    cache_tag = f"{id(res)}:{len(pp.df)}:{primary}"
    required = ("xlsx", "files", "zip", "parquet", "csv")
    ex = st.session_state.get("exports") or {}
    if ex.get("tag") != cache_tag or any(k not in ex for k in required):
        # Build into a fresh dict and commit only when complete, so an interrupted
        # rerun never leaves a half-built cache behind.
        new: dict = {}
        new["xlsx"] = excel_bytes(res, rec, primary, pp)
        new["files"] = model_files(res, primary)
        if pdf_available():
            pdf_ch = ch if ch else charts.build_pdf_charts(res)
            new["pdf"] = pdf_bytes(res, rec, primary, pdf_ch, pp)
        bundle = dict(new["files"])
        bundle[cfg.EXCEL_EVAL_NAME] = new["xlsx"]
        if "pdf" in new:
            bundle[cfg.PDF_REPORT_NAME] = new["pdf"]
        new["zip"] = zip_bytes(bundle)

        pq_disk = cfg.resolve_dataset_file("ookla_indonesia_only.parquet")
        csv_disk = cfg.resolve_dataset_file("ookla_indonesia_only.csv")
        if len(pp.df) == 608_990 and pq_disk.exists():
            new["parquet"] = pq_disk.read_bytes()
        else:
            new["parquet"] = pp_mod.to_parquet_bytes(pp.df)
        if len(pp.df) == 608_990 and csv_disk.exists():
            new["csv"] = csv_disk.read_bytes()
        else:
            new["csv"] = pp_mod.to_csv_bytes(pp.df)

        new["tag"] = cache_tag
        st.session_state["exports"] = new
        ex = new

    ui.section("Unduh Laporan & Model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            cfg.EXCEL_EVAL_NAME,
            ex["xlsx"],
            cfg.EXCEL_EVAL_NAME,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"{key_prefix}_dl_xlsx",
            on_click="ignore",
        )
    with c2:
        if "pdf" in ex:
            st.download_button(
                cfg.PDF_REPORT_NAME,
                ex["pdf"],
                cfg.PDF_REPORT_NAME,
                "application/pdf",
                width="stretch",
                key=f"{key_prefix}_dl_pdf",
                on_click="ignore",
            )
    with c3:
        st.download_button(
            cfg.ZIP_NAME,
            ex["zip"],
            cfg.ZIP_NAME,
            "application/zip",
            width="stretch",
            key=f"{key_prefix}_dl_zip",
            on_click="ignore",
        )

    ui.section("Unduh Dataset Indonesia Bersih")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "ookla_indonesia_only.parquet",
            ex["parquet"],
            "ookla_indonesia_only.parquet",
            "application/octet-stream",
            width="stretch",
            key=f"{key_prefix}_dl_pq",
            on_click="ignore",
        )
    with d2:
        st.download_button(
            "ookla_indonesia_only.csv",
            ex["csv"],
            "ookla_indonesia_only.csv",
            "text/csv",
            width="stretch",
            key=f"{key_prefix}_dl_csv",
            on_click="ignore",
        )

