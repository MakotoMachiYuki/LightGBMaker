"""Shared UI helpers: embedded fonts, dark-indigo HTML tables, KPI tiles, pills."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Fonts (@font-face base64, same approach as website/api.py)
# ══════════════════════════════════════════════════════════════════════════════
def _subset_ttf_b64(path: Path) -> str:
    try:
        from fontTools.subset import Options, Subsetter
        from fontTools.ttLib import TTFont

        font = TTFont(str(path))
        opt = Options()
        opt.layout_features = ["*"]
        sub = Subsetter(options=opt)
        sub.populate(unicodes=list(range(0x0020, 0x0250)) + list(range(0x2000, 0x2070))
                     + list(range(0x2190, 0x2200)) + list(range(0x2200, 0x2300)))
        sub.subset(font)
        buf = io.BytesIO()
        font.save(buf)
        raw = buf.getvalue()
    except Exception:
        raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


@st.cache_resource(show_spinner=False)
def font_face_css() -> str:
    blocks = []
    if cfg.FONT_YURUKA_PATH.exists():
        blocks.append(
            '@font-face{font-family:"FOT-Yuruka Std";'
            'src:local("FOT-Yuruka Std UB"),local("FOT-Yuruka Std"),local("YurukaStd-UB"),'
            f'url("data:font/ttf;base64,{_subset_ttf_b64(cfg.FONT_YURUKA_PATH)}") format("truetype");'
            "font-weight:100 900;font-style:normal;font-display:swap;}"
        )
    if cfg.FONT_NOTO_PATH.exists():
        blocks.append(
            '@font-face{font-family:"Noto Sans";src:local("Noto Sans"),'
            f'url("data:font/ttf;base64,{_subset_ttf_b64(cfg.FONT_NOTO_PATH)}") format("truetype");'
            "font-weight:100 900;font-style:normal;font-display:swap;}"
        )
    if cfg.FONT_NOTO_ITALIC_PATH.exists():
        blocks.append(
            '@font-face{font-family:"Noto Sans";src:local("Noto Sans Italic"),'
            f'url("data:font/ttf;base64,{_subset_ttf_b64(cfg.FONT_NOTO_ITALIC_PATH)}") format("truetype");'
            "font-weight:100 900;font-style:italic;font-display:swap;}"
        )
    return f"<style>{''.join(blocks)}</style>" if blocks else ""


# ══════════════════════════════════════════════════════════════════════════════
# Small HTML components
# ══════════════════════════════════════════════════════════════════════════════
def cls_pill(name: str, big: bool = False) -> str:
    color = cfg.QUALITY_COLORS.get(name, "#64748b")
    return f'<span class="dt-cls{" big" if big else ""}" style="--c:{color}">{html.escape(name)}</span>'


def split_badge(name: str) -> str:
    return f'<span class="split-badge" style="--c:{cfg.SPLIT_COLORS.get(name, "#4da8fa")}">{name}</span>'


def win_chip(name: str) -> str:
    if name == "tie":
        return '<span class="win-chip tie">Seri</span>'
    return f'<span class="win-chip" style="--c:{cfg.SPLIT_COLORS.get(name, "#4da8fa")}">{name}</span>'


def card_header(step: str | None, title: str, sub: str = "") -> None:
    badge = f'<span class="card-step">{html.escape(step)}</span>' if step else ""
    st.html(
        f'<div class="card-title">{badge}{html.escape(title)}</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else "")
    )


def render_recommendation(rec: dict, res: dict) -> None:
    """Render flat recommendation header + side-by-side Bahasa Indonesia comparison table."""
    winner = rec["winner"]
    scores_txt = " &nbsp;|&nbsp; ".join(f"{k}: <b>{v:.2f}</b>" for k, v in rec["scores"].items())
    crit = rec.get("criteria", [])
    names = [n for n in cfg.SPLIT_NAMES if n in res["splits"]]
    fmt_map = {
        "pct": lambda v: f"{v * 100:.2f}%",
        "f4": lambda v: f"{v:.4f}",
        "s": lambda v: f"{v:.2f} detik",
        "s4": lambda v: f"{v:.4f} detik",
        "ms4": lambda v: f"{v:.4f} ms",
    }
    head_cols = ["Metrik Evaluasi", "Kategori", "Bobot", *[f"Skema {n}" for n in names], "Kriteria Terbaik", "Model Unggul"]
    head_html = "".join(f"<th>{h}</th>" for h in head_cols)
    rows_html = ""
    for c in crit:
        f = fmt_map.get(c["fmt"], lambda v: str(v))
        cells = "".join(
            f'<td class="{"best" if c["Model Unggul"] == n else ""}">{f(c[n])}</td>'
            for n in names
        )
        rows_html += (
            f"<tr>"
            f"<th>{html.escape(c['Metrik Evaluasi'])}</th>"
            f"<td>{html.escape(c['Kategori'])}</td>"
            f"<td>{c['Bobot']:.2f}</td>"
            f"{cells}"
            f"<td>{html.escape(c['Kriteria Terbaik'])}</td>"
            f"<td>{win_chip(c['Model Unggul'])}</td>"
            f"</tr>"
        )
    # Summary score row at the bottom of the table
    score_cells = "".join(
        f'<td class="{"best" if winner == n else ""}"><b>{rec["scores"].get(n, 0.0):.2f}</b></td>'
        for n in names
    )
    total_w = sum(c["Bobot"] for c in crit)
    rows_html += (
        f'<tr class="dt-you">'
        f"<th>Total Skor Evaluasi</th>"
        f"<td>Akumulasi Bobot</td>"
        f"<td>{total_w:.2f}</td>"
        f"{score_cells}"
        f"<td>Skor Tertinggi</td>"
        f"<td>{win_chip(winner)}</td>"
        f"</tr>"
    )

    st.html(
        f"""
        <div class="rec-banner">
          <div class="rec-top">
            <div>
              <div class="rec-kicker">REKOMENDASI MODEL OTOMATIS</div>
              <div class="rec-title">Model Terbaik: <span class="rec-win">{winner}</span></div>
            </div>
            <div class="rec-score">Skor Evaluasi<br>{scores_txt}</div>
          </div>
          <div class="dt-wrap" style="margin-top:0.85rem">
            <table class="dt-table">
              <thead><tr>{head_html}</tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>
        """
    )


def section(title: str) -> None:
    st.html(f'<div class="sec-title">{html.escape(title)}</div>')


def kpis(items: list[tuple]) -> None:
    """items: (label, value, sub, variant)"""
    cells = []
    for it in items:
        label, value, sub, variant = (list(it) + ["", ""])[:4]
        sub_html = f'<div class="kpi-s">{sub}</div>' if sub else ""
        cells.append(
            f'<div class="kpi {variant}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div>{sub_html}</div>'
        )
    st.html(f'<div class="kpi-grid">{"".join(cells)}</div>')


def _fmt_cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "Ya" if v else "Tidak"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if abs(fv) >= 1000:
            return f"{fv:,.2f}"
        if fv.is_integer() and abs(fv) < 1000 and abs(fv) > 1:
            return f"{fv:,.1f}"
        return f"{fv:.4f}"
    s = str(v)
    for cname in cfg.CLASS_NAMES:  # render class labels as pills
        if s in (cname, f"{cfg.LABEL_MAPPING[cname]} - {cname}"):
            return cls_pill(s)
    if s in cfg.SPLIT_COLORS:
        return split_badge(s)
    return html.escape(s)


def html_table(
    df: pd.DataFrame,
    title: str | None = None,
    subtitle: str = "",
    index: bool = False,
    you_rows: set | None = None,
    you_col: str | None = None,
    best_cols: dict | None = None,
    diag: bool = False,
    hl_cols: set | None = None,
) -> str:
    """Render a DataFrame as a dark-indigo `.dt-table`.

    you_rows/you_col: highlight rows whose `you_col` value is in `you_rows`.
    best_cols: {col: 'max'|'min'} -> green-highlight the best value per column.
    diag: highlight the diagonal (confusion matrices).
    """
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" · ".join(str(x) for x in c) for c in d.columns]
    if index:
        d = d.reset_index()
    best_vals = {}
    for c, mode in (best_cols or {}).items():
        if c in d.columns:
            col = pd.to_numeric(d[c], errors="coerce")
            if col.notna().sum() > 1 and col.nunique() > 1:
                best_vals[c] = col.max() if mode == "max" else col.min()
    head = "".join(
        f'<th class="{"hl-col" if hl_cols and c in hl_cols else ""}">{html.escape(str(c))}</th>'
        for c in d.columns
    )
    rows = []
    for ri, (_, r) in enumerate(d.iterrows()):
        cls = ""
        if you_rows and you_col and str(r.get(you_col)) in you_rows:
            cls = ' class="dt-you"'
        cells = []
        for ci, c in enumerate(d.columns):
            v = r[c]
            tag = "th" if ci == 0 else "td"
            klass = []
            if c in best_vals:
                try:
                    if float(v) == float(best_vals[c]):
                        klass.append("best")
                except Exception:
                    pass
            if diag and ci > 0 and ci - 1 == ri:
                klass.append("diag")
            if hl_cols and c in hl_cols:
                klass.append("hl-col")
            ka = f' class="{" ".join(klass)}"' if klass else ""
            cells.append(f"<{tag}{ka}>{_fmt_cell(v)}</{tag}>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    table = (
        f'<div class="dt-wrap"><table class="dt-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )
    if title:
        return (
            f'<div class="dt-section"><div class="dt-section-head"><h4>{html.escape(title)}</h4>'
            f"<span>{subtitle}</span></div>{table}</div>"
        )
    return f'<div class="dt-section">{table}</div>'


def table(df: pd.DataFrame, title: str | None = None, **kw) -> None:
    st.html(html_table(df, title, **kw))
