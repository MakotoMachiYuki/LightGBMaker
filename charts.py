"""Native interactive Plotly charts for the Streamlit UI + Matplotlib PNGs for PDF export.

All UI charts in `Home` and `Classification` are rendered natively as interactive
Plotly figures (`st.plotly_chart`) rather than static images. Matplotlib helpers
are kept strictly for embedding charts inside the downloadable PDF report.
"""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import config as cfg  # noqa: E402

SPLIT_COL = cfg.SPLIT_COLORS
CLASS_COLS = [cfg.QUALITY_COLORS[c] for c in cfg.CLASS_NAMES]
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _apply_plotly_theme(fig: go.Figure, title: str, height: int = 390, has_subplots: bool = False) -> go.Figure:
    top_margin = 88 if has_subplots else 68
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"family": "FOT-Yuruka Std, Noto Sans, sans-serif", "size": 17, "color": cfg.INK},
            "x": 0.01,
            "y": 0.96,
            "xanchor": "left",
            "yanchor": "top",
        },
        font={"family": "Noto Sans, sans-serif", "size": 12, "color": cfg.INK},
        paper_bgcolor="rgba(255, 255, 255, 0.92)",
        plot_bgcolor="#f6fbff",
        height=height,
        margin={"l": 56, "r": 28, "t": top_margin, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08 if has_subplots else 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(255,255,255,0.75)",
            "font": {"size": 11.5, "color": cfg.INK},
        },
        hoverlabel={"font": {"family": "Noto Sans, sans-serif", "size": 12}},
    )
    if has_subplots and fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.font = {"family": "FOT-Yuruka Std, Noto Sans, sans-serif", "size": 13, "color": cfg.INK}
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(11, 37, 69, 0.10)",
        linecolor=cfg.INK,
        tickfont={"color": cfg.INK},
        title_font={"color": cfg.INK},
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(11, 37, 69, 0.12)",
        linecolor=cfg.INK,
        tickfont={"color": cfg.INK},
        title_font={"color": cfg.INK},
        zeroline=False,
        automargin=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Native Interactive Plotly Figures (for Streamlit UI)
# ══════════════════════════════════════════════════════════════════════════════
def fig_class_distribution(dist_df: pd.DataFrame) -> go.Figure:
    counts = dist_df["Jumlah Sampel"].tolist()
    pcts = dist_df["Persentase (%)"].tolist()
    labels = [f"{c:,} ({p:.2f}%)" for c, p in zip(counts, pcts)]
    fig = go.Figure(
        go.Bar(
            x=dist_df["Kelas Kualitas"],
            y=counts,
            marker={"color": CLASS_COLS, "line": {"color": cfg.INK, "width": 1.5}},
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Jumlah Sampel: %{y:,}<extra></extra>",
        )
    )
    _apply_plotly_theme(fig, "Distribusi Kelas Kualitas Jaringan", height=400)
    fig.update_yaxes(title_text="Jumlah Sampel", range=[0, max(counts) * 1.22 + 1])
    fig.update_xaxes(title_text="Kelas Kualitas")
    return fig


def fig_top_provinces(prov_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    d = prov_df.head(top_n).iloc[::-1]
    fig = go.Figure()
    for cname in cfg.CLASS_NAMES:
        fig.add_trace(
            go.Bar(
                y=d["Provinsi"],
                x=d[cname],
                name=cname,
                orientation="h",
                marker={"color": cfg.QUALITY_COLORS[cname], "line": {"color": "#ffffff", "width": 0.6}},
                hovertemplate=f"<b>%{{y}}</b> · {cname}: %{{x:,}}<extra></extra>",
            )
        )
    _apply_plotly_theme(fig, f"Top {len(d)} Provinsi Berdasarkan Kualitas", height=400)
    fig.update_layout(
        barmode="stack",
        margin={"l": 150, "r": 24, "t": 84, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(title_text="Jumlah Sampel")
    return fig


def fig_global_comparison(results: dict) -> go.Figure:
    comp = results["comparison_df"]
    metrics = [
        "Overall Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-Score",
    ]
    fig = go.Figure()
    for _, row in comp.iterrows():
        sname = row["Skema Split"]
        vals = [float(row[m]) for m in metrics]
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=vals,
                name=f"Skema {sname}",
                marker={"color": SPLIT_COL.get(sname, "#4da8fa"), "line": {"color": cfg.INK, "width": 1.2}},
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{x}}</b> ({sname}): %{{y:.4f}}<extra></extra>",
            )
        )
    lo = float(comp[metrics].min().min())
    _apply_plotly_theme(fig, "Perbandingan Metrik Evaluasi Global (Data Uji)", height=410)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Skor Metrik", range=[max(0.0, lo - 0.08), 1.06])
    return fig


def fig_per_class(results: dict) -> go.Figure:
    sp = results["splits"]
    names = list(sp.keys())
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[f"Skema {n}" for n in names],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    mets = [
        ("Precision (P_k)", "#4da8fa"),
        ("Recall (R_k)", "#7c3aed"),
        ("F1-Score (F1_k)", "#f1b6fb"),
    ]
    for col_idx, sname in enumerate(names, start=1):
        ovr = sp[sname]["ovr_df"]
        for m_col, color in mets:
            vals = [float(v) for v in ovr[m_col]]
            fig.add_trace(
                go.Bar(
                    x=ovr["Kelas (k)"],
                    y=vals,
                    name=m_col,
                    legendgroup=m_col,
                    showlegend=(col_idx == 1),
                    marker={"color": color, "line": {"color": cfg.INK, "width": 1.0}},
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=f"<b>%{{x}}</b> · {m_col} ({sname}): %{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
    _apply_plotly_theme(fig, "Evaluasi Precision, Recall, dan F1-Score per Kelas", height=420, has_subplots=True)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.16], title_text="Skor", row=1, col=1)
    return fig


def fig_feature_importance(results: dict) -> go.Figure:
    sp = results["splits"]
    names = list(sp.keys())
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[f"Skema {n}" for n in names],
        horizontal_spacing=0.12,
    )
    for col_idx, sname in enumerate(names, start=1):
        fi = sp[sname]["fi_df"].sort_values("Gain Importance", ascending=True)
        vals = [float(v) for v in fi["Gain Importance"]]
        fig.add_trace(
            go.Bar(
                y=fi["Fitur"],
                x=vals,
                orientation="h",
                name=f"Gain ({sname})",
                showlegend=False,
                marker={"color": SPLIT_COL.get(sname, "#4da8fa"), "line": {"color": cfg.INK, "width": 1.2}},
                text=[f"{v:,.0f}" for v in vals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{y}}</b> ({sname})<br>Gain Importance: %{{x:,.2f}}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(range=[0, max(vals) * 1.28 + 1], title_text="Total Information Gain", row=1, col=col_idx)
    _apply_plotly_theme(fig, "Tingkat Kepentingan Fitur (Feature Importance — Gain)", height=390, has_subplots=True)
    fig.update_layout(margin={"l": 125, "r": 36, "t": 84, "b": 52})
    return fig


def fig_confusion_matrices(results: dict) -> go.Figure:
    sp = results["splits"]
    names = list(sp.keys())
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[
            f"Skema {n} (Akurasi: {sp[n]['accuracy'] * 100:.2f}%)"
            for n in names
        ],
        horizontal_spacing=0.10,
    )
    scales = {"80:20": "Blues", "70:30": "Purples"}
    for col_idx, sname in enumerate(names, start=1):
        cm = sp[sname]["cm"]
        cm_flip = cm[::-1]
        y_labels = cfg.CLASS_NAMES[::-1]
        text_grid = [[f"{int(v):,}" for v in row] for row in cm_flip]
        fig.add_trace(
            go.Heatmap(
                z=cm_flip,
                x=cfg.CLASS_NAMES,
                y=y_labels,
                text=text_grid,
                texttemplate="<b>%{text}</b>",
                colorscale=scales.get(sname, "Blues"),
                showscale=False,
                hovertemplate="Aktual: <b>%{y}</b><br>Prediksi: <b>%{x}</b><br>Jumlah: <b>%{z:,}</b><extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(title_text="Prediksi Model", row=1, col=col_idx)
        fig.update_yaxes(title_text="Kelas Aktual", row=1, col=col_idx)
    _apply_plotly_theme(fig, "Confusion Matrix 5×5 (Skema 80:20 vs 70:30)", height=440, has_subplots=True)
    fig.update_layout(margin={"l": 95, "r": 28, "t": 84, "b": 55})
    return fig


def fig_logloss_curves(results: dict) -> go.Figure:
    sp = results["splits"]
    names = list(sp.keys())
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[
            f"Skema {n} ({sp[n]['train_time_sec']:.2f}s)"
            for n in names
        ],
        shared_yaxes=True,
        horizontal_spacing=0.07,
    )
    for col_idx, sname in enumerate(names, start=1):
        r = sp[sname]
        ev = r.get("evals") or {}
        tr = ev.get("Train", {}).get("multi_logloss", [])
        va = ev.get("Validation", {}).get("multi_logloss", [])
        iters = list(range(1, len(tr) + 1))
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=tr,
                mode="lines",
                name="Train Multi-Logloss",
                legendgroup="train",
                showlegend=(col_idx == 1),
                line={"color": SPLIT_COL.get(sname, "#4da8fa"), "width": 2.4},
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(va) + 1)),
                y=va,
                mode="lines",
                name="Validation Multi-Logloss",
                legendgroup="val",
                showlegend=(col_idx == 1),
                line={"color": "#e11d48", "width": 2.4, "dash": "dash"},
            ),
            row=1,
            col=col_idx,
        )
        bi = r.get("best_iteration")
        if bi:
            fig.add_vline(
                x=bi,
                line_width=1.6,
                line_dash="dot",
                line_color=cfg.INK,
                annotation_text=f"Best ({bi})",
                annotation_position="top left",
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="Iterasi Pohon (Boosting Rounds)", row=1, col=col_idx)
    fig.update_yaxes(title_text="Multi-Logloss", row=1, col=1)
    _apply_plotly_theme(fig, "Konvergensi Multi-Logloss (Training vs Validation)", height=390, has_subplots=True)
    return fig


def fig_kfold_lines(results: dict) -> go.Figure:
    sp = results["splits"]
    metrics = [
        ("Accuracy (E_i)", "Akurasi per Fold"),
        ("Macro F1-Score", "Macro F1 per Fold"),
        ("Weighted F1-Score", "Weighted F1 per Fold"),
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.09,
    )
    for col_idx, (col_name, _) in enumerate(metrics, start=1):
        for sname, r in sp.items():
            cv = r.get("cv_df")
            if cv is None:
                continue
            folds = cv[cv["Fold"].astype(str).str.startswith("Fold-")]
            fig.add_trace(
                go.Scatter(
                    x=folds["Fold"],
                    y=folds[col_name],
                    mode="lines+markers",
                    name=f"Skema {sname}",
                    legendgroup=sname,
                    showlegend=(col_idx == 1),
                    line={"color": SPLIT_COL.get(sname, "#4da8fa"), "width": 2.4},
                    marker={"size": 8},
                    hovertemplate=f"<b>%{{x}}</b> ({sname}): %{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
    _apply_plotly_theme(fig, "Evaluasi K-Fold Cross Validation per Fold", height=390, has_subplots=True)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib PNG Helpers (Strictly for PDF Report Export)
# ══════════════════════════════════════════════════════════════════════════════
DPI = 160
TITLE_FONT = "DejaVu Sans"
BODY_FONT = "DejaVu Sans"


def _register_fonts() -> None:
    global TITLE_FONT, BODY_FONT
    for path in (cfg.FONT_YURUKA_PATH, cfg.FONT_NOTO_PATH):
        try:
            if path.exists():
                font_manager.fontManager.addfont(str(path))
        except Exception:
            pass
    try:
        if cfg.FONT_YURUKA_PATH.exists():
            TITLE_FONT = font_manager.FontProperties(fname=str(cfg.FONT_YURUKA_PATH)).get_name()
        if cfg.FONT_NOTO_PATH.exists():
            BODY_FONT = font_manager.FontProperties(fname=str(cfg.FONT_NOTO_PATH)).get_name()
    except Exception:
        pass


_register_fonts()

plt.rcParams.update({
    "font.family": [BODY_FONT, "DejaVu Sans"],
    "axes.edgecolor": cfg.INK,
    "axes.labelcolor": cfg.INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.color": cfg.INK,
    "ytick.color": cfg.INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": cfg.INK,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "#f6fbff",
})


def _title(ax, text: str, size: float = 12.5) -> None:
    ax.set_title(text, fontfamily=[TITLE_FONT, "DejaVu Sans"], fontsize=size,
                 color=cfg.INK, pad=10)


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, fontfamily=[TITLE_FONT, "DejaVu Sans"], fontsize=14.5,
                 color=cfg.INK, y=1.02)


def to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _bar_labels(ax, bars, fmt: str = "{:.4f}", size: float = 7.5, rot: int = 0) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=size, rotation=rot,
                        xytext=(0, 2), textcoords="offset points", color=cfg.INK)


def class_distribution(dist_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bars = ax.bar(dist_df["Kelas Kualitas"], dist_df["Jumlah Sampel"], color=CLASS_COLS,
                  edgecolor=cfg.INK, linewidth=1.2)
    for b, pct in zip(bars, dist_df["Persentase (%)"]):
        ax.annotate(f"{int(b.get_height()):,}\n({pct:.2f}%)",
                    (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    ax.set_ylabel("Jumlah Sampel")
    ax.set_ylim(0, dist_df["Jumlah Sampel"].max() * 1.22 + 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _title(ax, "Distribusi Kelas Kualitas Jaringan")
    return to_png(fig)


def top_provinces(prov_df: pd.DataFrame, top_n: int = 15) -> bytes:
    d = prov_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.36 * len(d) + 1)))
    left = np.zeros(len(d))
    for cname in cfg.CLASS_NAMES:
        vals = d[cname].to_numpy()
        ax.barh(d["Provinsi"], vals, left=left, color=cfg.QUALITY_COLORS[cname],
                edgecolor="white", linewidth=0.5, label=cname)
        left += vals
    for y, tot in enumerate(d["Total"]):
        ax.text(tot, y, f" {int(tot):,}", va="center", fontsize=8)
    ax.set_xlabel("Jumlah Sampel")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(0, d["Total"].max() * 1.15 + 1)
    ax.legend(ncol=5, loc="lower right", fontsize=8)
    _title(ax, f"Top {len(d)} Provinsi berdasarkan Jumlah Sampel × Kualitas")
    return to_png(fig)


def logloss_curves(results: dict) -> bytes:
    sp = results["splits"]
    fig, axes = plt.subplots(1, len(sp), figsize=(7 * len(sp), 4.2), sharey=True, squeeze=False)
    for ax, (name, r) in zip(axes[0], sp.items()):
        ev = r["evals"]
        tr = ev.get("Train", {}).get("multi_logloss", [])
        va = ev.get("Validation", {}).get("multi_logloss", [])
        ax.plot(tr, label="Training Multi-Logloss", lw=2.2, color=SPLIT_COL[name])
        ax.plot(va, label="Validation Multi-Logloss", lw=2.2, ls="--", color="#e11d48")
        bi = r["best_iteration"]
        ax.axvline(bi - 1 if bi else 0, color=cfg.INK, ls=":", lw=1.6, label=f"Best Iter ({bi})")
        _title(ax, f"Multi-Logloss Skema {name} (Waktu: {r['train_time_sec']:.2f}s)")
        ax.set_xlabel("Iterasi Pohon (Boosting Rounds)")
        ax.set_ylabel("Multi-Logloss")
        ax.legend()
    fig.tight_layout()
    return to_png(fig)


def kfold_lines(results: dict) -> bytes:
    sp = results["splits"]
    metrics = [("Accuracy (E_i)", "Accuracy"), ("Macro F1-Score", "Macro F1-Score"),
               ("Weighted F1-Score", "Weighted F1-Score")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, (col, label) in zip(axes, metrics):
        for name, r in sp.items():
            cv = r.get("cv_df")
            if cv is None:
                continue
            folds = cv[cv["Fold"].str.startswith("Fold-")]
            ax.plot(folds["Fold"], folds[col], marker="o", lw=2.2, ms=7,
                    color=SPLIT_COL[name], label=name)
            mean = cv.loc[cv["Fold"] == "Rata-rata (CV)", col].iloc[0]
            ax.axhline(mean, color=SPLIT_COL[name], ls=":", lw=1.3, alpha=0.9)
        _title(ax, f"{label} per Fold", 11.5)
        ax.legend(title="Skema")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
    _suptitle(fig, "K-Fold Cross Validation")
    fig.tight_layout()
    return to_png(fig)


def kfold_bars(results: dict) -> bytes:
    sp = {k: v for k, v in results["splits"].items() if v.get("cv_df") is not None}
    fig, axes = plt.subplots(1, len(sp), figsize=(7 * len(sp), 3.9), squeeze=False)
    for ax, (name, r) in zip(axes[0], sp.items()):
        cv = r["cv_df"]
        folds = cv[cv["Fold"].str.startswith("Fold-")]
        x = np.arange(len(folds))
        w = 0.38
        b1 = ax.bar(x - w / 2, folds["Accuracy (E_i)"], w, label="Accuracy",
                    color=SPLIT_COL[name], edgecolor=cfg.INK)
        b2 = ax.bar(x + w / 2, folds["Macro F1-Score"], w, label="Macro F1",
                    color="#f1b6fb", edgecolor=cfg.INK)
        _bar_labels(ax, b1, "{:.4f}", 7, 90)
        _bar_labels(ax, b2, "{:.4f}", 7, 90)
        ax.set_xticks(x, folds["Fold"])
        lo = min(folds["Macro F1-Score"].min(), folds["Accuracy (E_i)"].min())
        ax.set_ylim(max(0, lo - 0.08), 1.04)
        _title(ax, f"K-Fold Accuracy & Macro F1 — Skema {name}", 11.5)
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    fig.tight_layout()
    return to_png(fig)


def global_comparison(results: dict) -> bytes:
    comp = results["comparison_df"]
    metrics = ["Overall Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score",
               "Weighted Precision", "Weighted Recall", "Weighted F1-Score"]
    fig, ax = plt.subplots(figsize=(12, 4.3))
    x = np.arange(len(metrics))
    n = len(comp)
    w = 0.8 / max(n, 1)
    for i, (_, row) in enumerate(comp.iterrows()):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, [row[m] for m in metrics], w,
                      label=row["Skema Split"], color=SPLIT_COL[row["Skema Split"]],
                      edgecolor=cfg.INK)
        _bar_labels(ax, bars, "{:.4f}", 7.5)
    ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in metrics], fontsize=8.5)
    lo = comp[metrics].min().min()
    ax.set_ylim(max(0, lo - 0.1), 1.03)
    ax.legend(title="Skema Split")
    _title(ax, "Perbandingan Metrik Evaluasi Global (Data Uji)")
    return to_png(fig)


def time_comparison(results: dict) -> bytes:
    comp = results["comparison_df"]
    sp = results["splits"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    names = list(comp["Skema Split"])
    cols = [SPLIT_COL[n] for n in names]
    panels = [
        ("Waktu Training (detik)", comp["Waktu Training (detik)"], "{:.2f}s"),
        ("Waktu Prediksi Test Set (detik)", comp["Waktu Prediksi (detik)"], "{:.3f}s"),
        ("Latensi 1 Record (ms)", [sp[n]["latency"]["single_ms"] for n in names], "{:.2f}"),
    ]
    for ax, (ttl, vals, fmt) in zip(axes, panels):
        vals = list(vals)
        bars = ax.bar(names, vals, color=cols, edgecolor=cfg.INK)
        _bar_labels(ax, bars, fmt, 9)
        ax.set_ylim(0, max(vals) * 1.25 + 1e-6)
        _title(ax, ttl, 11)
    _suptitle(fig, "Waktu Komputasi: Training & Inferensi")
    fig.tight_layout()
    return to_png(fig)


def per_class(results: dict) -> bytes:
    sp = results["splits"]
    fig, axes = plt.subplots(1, len(sp), figsize=(7.2 * len(sp), 4), sharey=True, squeeze=False)
    mets = [("Precision (P_k)", "#4da8fa"), ("Recall (R_k)", "#7c3aed"), ("F1-Score (F1_k)", "#f1b6fb")]
    for ax, (name, r) in zip(axes[0], sp.items()):
        ovr = r["ovr_df"]
        x = np.arange(len(ovr))
        w = 0.27
        for i, (m, c) in enumerate(mets):
            bars = ax.bar(x + (i - 1) * w, ovr[m], w, label=m, color=c, edgecolor=cfg.INK)
            _bar_labels(ax, bars, "{:.3f}", 6.5, 90)
        ax.set_xticks(x, ovr["Kelas (k)"], fontsize=8.5)
        ax.set_ylim(0, 1.15)
        _title(ax, f"Metrik per Kelas (OVR) — Skema {name}", 11.5)
        ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return to_png(fig)


def confusion_matrices(results: dict) -> bytes:
    sp = results["splits"]
    fig, axes = plt.subplots(1, len(sp), figsize=(7.5 * len(sp), 5.8), squeeze=False)
    cmaps = {"80:20": "Blues", "70:30": "Purples"}
    for ax, (name, r) in zip(axes[0], sp.items()):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap=cmaps.get(name, "Blues"),
                    xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax,
                    linewidths=0.6, linecolor="white", cbar_kws={"shrink": 0.8},
                    annot_kws={"fontsize": 9})
        _title(ax, f"Confusion Matrix Skema {name} (Acc: {r['accuracy'] * 100:.2f}%)")
        ax.set_xlabel("Prediksi Model (Predicted Label)")
        ax.set_ylabel("Kelas Aktual (True Label)")
        ax.grid(False)
    fig.tight_layout()
    return to_png(fig)


def feature_importance(results: dict) -> bytes:
    sp = results["splits"]
    fig, axes = plt.subplots(2, len(sp), figsize=(7 * len(sp), 7.2), squeeze=False)
    for j, (name, r) in enumerate(sp.items()):
        fi = r["fi_df"]
        for i, (col, pal) in enumerate([("Gain Importance", "viridis"), ("Split Importance", "magma")]):
            ax = axes[i][j]
            d = fi.sort_values(col, ascending=True)
            colors = sns.color_palette(pal, len(d))
            bars = ax.barh(d["Fitur"], d[col], color=colors, edgecolor=cfg.INK)
            for b in bars:
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                        f" {b.get_width():,.0f}", va="center", fontsize=7.5)
            ax.set_xlim(0, d[col].max() * 1.22 + 1)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            _title(ax, f"{col} — Skema {name}", 11)
    fig.tight_layout()
    return to_png(fig)


def split_distribution(results: dict) -> bytes:
    d = results["split_summary_df"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels = cfg.CLASS_LABELS
    x = np.arange(len(labels))
    combos = [(s, kind) for s in results["splits"] for kind in ("Train Count", "Test Count")]
    w = 0.8 / len(combos)
    for i, (s, kind) in enumerate(combos):
        vals = [int(d[(d["Skema Split"] == s) & (d["Kelas Kualitas"] == l)][kind].iloc[0]) for l in labels]
        ax.bar(x + (i - (len(combos) - 1) / 2) * w, vals, w, label=f"{s} {kind.split()[0]}",
               color=SPLIT_COL[s], alpha=1.0 if kind == "Train Count" else 0.5, edgecolor=cfg.INK)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("Jumlah (log)")
    ax.legend(ncol=4, fontsize=8)
    _title(ax, "Distribusi Train/Test per Kelas (Stratified)")
    return to_png(fig)


def build_pdf_charts(res: dict) -> dict[str, bytes]:
    """Generate PNG bytes strictly when building the downloadable PDF report."""
    out = {
        "logloss": logloss_curves(res),
        "global": global_comparison(res),
        "time": time_comparison(res),
        "per_class": per_class(res),
        "cm": confusion_matrices(res),
        "fi": feature_importance(res),
    }
    if any(r.get("cv_df") is not None for r in res["splits"].values()):
        out["kfold_line"] = kfold_lines(res)
        out["kfold_bar"] = kfold_bars(res)
    return out
