"""Home view — Ringkasan Hasil Klasifikasi yang Telah Dilakukan.

Menampilkan ringkasan klasifikasi menggunakan kartu metrik utama, tabel
perbandingan rekomendasi model, peta sebaran kualitas jaringan Indonesia,
serta tab Grafik (native interactive charts), Confusion Matrix, Evaluasi, dan Export.
"""

from __future__ import annotations

import html

import streamlit as st

import charts
import exporter
import maps
import modelling
import ui


@st.cache_resource(show_spinner=False)
def get_default_bundle() -> dict:
    """Load pre-computed workspace results once."""
    pp, res, rec = modelling.load_precomputed_workspace()
    if pp is None or res is None:
        return {}
    return {"pp": pp, "results": res, "rec": rec}


def ensure_state_loaded() -> None:
    """Populate session state from the pre-trained workspace if not already set."""
    if st.session_state.get("results") is not None and st.session_state.get("pp") is not None:
        return
    default = get_default_bundle()
    if not default:
        return
    st.session_state.pp = default["pp"]
    st.session_state.results = default["results"]
    st.session_state.rec = default["rec"]


def render() -> None:
    ensure_state_loaded()
    pp = st.session_state.get("pp")
    res = st.session_state.get("results")
    rec = st.session_state.get("rec")

    if pp is None or res is None or rec is None:
        with st.container(key="card-home-empty"):
            ui.card_header(None, "Ringkasan Klasifikasi", "Belum ada data klasifikasi yang tersedia.")
            st.info("Buka halaman Classification untuk menjalankan klasifikasi data.")
        return

    sp = res["splits"]
    s = pp.stats
    winner = rec["winner"]
    years_str = "–".join([str(s["years"][0]), str(s["years"][-1])]) if len(s.get("years", [])) > 1 else (
        str(s["years"][0]) if s.get("years") else "Semua Tahun"
    )

    # ── 1. Summary KPI Card & Side-by-Side Model Recommendation Table ─────────
    with st.container(key="card-home-summary"):
        ui.card_header(
            None,
            "Ringkasan Hasil Klasifikasi Kualitas Jaringan Indonesia",
        )
        kpi_items = [
            ("Total Data Indonesia", f"{s['final_rows']:,}", f"{s['provinces']} Provinsi ({years_str})", "purple"),
            ("Model Terbaik", winner, "", "good"),
        ]
        for sname, r in sp.items():
            kpi_items.append(
                (f"Akurasi Uji ({sname})", f"{r['accuracy'] * 100:.2f}%",
                 f"Macro F1: {r['macro_f1']:.4f}", "good" if sname == winner else "")
            )
        cvo = res.get("cv_overview_df")
        if cvo is not None and not cvo.empty:
            best_cv = cvo["Mean CV Accuracy"].max()
            kpi_items.append(
                (f"Rata-rata {res['config']['k_folds']}-Fold CV", f"{best_cv * 100:.2f}%",
                 "", "")
            )
        ui.kpis(kpi_items)
        ui.render_recommendation(rec, res)

    # ── 2. Interactive Geospatial Map Card ────────────────────────────────────
    with st.container(key="card-home-map"):
        ui.card_header(
            None,
            "Peta Sebaran Kualitas Jaringan Seluler di Indonesia",
        )
        maps.render_map_section(pp.df, key_prefix="home")

    # ── 3. Evaluation Tabs: Grafik | Confusion Matrix | Evaluasi | Export ─────
    with st.container(key="card-home-visuals"):
        ui.card_header(
            None,
            "Grafik & Tabel Evaluasi Model LightGBM",
        )
        tabs = st.tabs(["Grafik", "Confusion Matrix", "Evaluasi", "Export"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    charts.fig_class_distribution(pp.dist_df),
                    width="stretch",
                    config=charts.PLOTLY_CONFIG,
                    key="home_fig_dist",
                )
            with c2:
                if not pp.prov_quality_df.empty:
                    st.plotly_chart(
                        charts.fig_top_provinces(pp.prov_quality_df, 15),
                        width="stretch",
                        config=charts.PLOTLY_CONFIG,
                        key="home_fig_prov",
                    )
            st.plotly_chart(
                charts.fig_global_comparison(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="home_fig_global",
            )
            st.plotly_chart(
                charts.fig_per_class(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="home_fig_per_class",
            )
            st.plotly_chart(
                charts.fig_feature_importance(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="home_fig_fi",
            )

        with tabs[1]:
            st.plotly_chart(
                charts.fig_confusion_matrices(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="home_fig_cm",
            )
            for sname, r in sp.items():
                ui.table(r["cm_df"], f"Tabel Confusion Matrix — {sname}", index=True, diag=True)
            st.plotly_chart(
                charts.fig_logloss_curves(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="home_fig_logloss",
            )
            if any(r.get("cv_df") is not None for r in sp.values()):
                st.plotly_chart(
                    charts.fig_kfold_lines(res),
                    width="stretch",
                    config=charts.PLOTLY_CONFIG,
                    key="home_fig_kfold",
                )

        with tabs[2]:
            ui.table(
                res["comparison_df"],
                "Perbandingan Performa Global (80:20 vs 70:30)",
                best_cols={
                    "Overall Accuracy": "max",
                    "Macro Precision": "max",
                    "Macro Recall": "max",
                    "Macro F1-Score": "max",
                    "Weighted F1-Score": "max",
                    "Waktu Training (detik)": "min",
                },
            )
            for sname, r in sp.items():
                ui.table(r["ovr_df"], f"Metrik per Kelas (OVR) — {sname}")
            if cvo is not None and not cvo.empty:
                ui.table(
                    cvo,
                    f"Ringkasan {res['config']['k_folds']}-Fold Cross Validation",
                    best_cols={"Mean CV Accuracy": "max", "Std CV Accuracy": "min", "Mean Macro F1-Score": "max"},
                )
            ui.table(pp.dist_df, "Distribusi Kelas Kualitas")
            if not pp.prov_quality_df.empty:
                ui.table(pp.prov_quality_df.head(15), "Top 15 Provinsi × Kualitas", best_cols={"Total": "max"})

        with tabs[3]:
            exporter.render_export_panel(pp, res, rec, st.session_state.get("charts") or {}, key_prefix="home")
