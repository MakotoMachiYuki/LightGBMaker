"""Classification view — Alur Data → Classification → Result.

* Tahap 1 · Data: Pilih atau upload file dataset (Bounding Box Indonesia, Spatial
  Join Poligon GDB 38 Provinsi, dan deteksi semua tahun aktif secara otomatis).
* Tahap 2 · Classification: Eksekusi satu klik menggunakan parameter default
  LightGBM (pembagian data 80:20 & 70:30 + 5-Fold Stratified CV).
* Tahap 3 · Result: Ditampilkan langsung di bagian bawah setelah pemodelan
  selesai, mencakup metrik utama, tabel perbandingan rekomendasi model, peta
  sebaran Indonesia, serta tab Grafik, Confusion Matrix, Evaluasi, dan Export.
"""

from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd
import streamlit as st

import charts
import config as cfg
import exporter
import home
import maps
import modelling
import preprocess as pp_mod
import ui


def _progress_cb(bar, text_el):
    def cb(p: float, msg: str) -> None:
        bar.progress(min(max(p, 0.0), 1.0))
        text_el.caption(msg)
    return cb


def _build_all_charts(pp, res: dict) -> tuple[dict, dict]:
    ch = {
        "logloss": charts.logloss_curves(res),
        "global": charts.global_comparison(res),
        "time": charts.time_comparison(res),
        "per_class": charts.per_class(res),
        "cm": charts.confusion_matrices(res),
        "fi": charts.feature_importance(res),
        "split_dist": charts.split_distribution(res),
    }
    if any(r.get("cv_df") is not None for r in res["splits"].values()):
        ch["kfold_line"] = charts.kfold_lines(res)
        ch["kfold_bar"] = charts.kfold_bars(res)
    pp_ch = {
        "dist": charts.class_distribution(pp.dist_df),
    }
    if not pp.prov_quality_df.empty:
        pp_ch["prov"] = charts.top_provinces(pp.prov_quality_df, 15)
    return ch, pp_ch


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 & 2 — Data & Classification Trigger
# ══════════════════════════════════════════════════════════════════════════════
def _render_data_and_run() -> None:
    with st.container(key="card-clf-data"):
        ui.card_header(
            "1",
            "Data — Pilih atau Upload Dataset Jaringan",
        )

        src_mode = st.radio(
            "Sumber Data",
            ["Dataset Tersedia (Siap Pakai)", "Upload File (.parquet / .csv / .xlsx)"],
            horizontal=True,
            key="clf_src_mode",
        )

        selections: list[tuple[str, object, str, int | None]] = []
        if src_mode.startswith("Upload"):
            ups = st.file_uploader(
                "Upload satu atau beberapa file dataset (.parquet, .csv, .xlsx)",
                type=["parquet", "xlsx", "xls", "csv"],
                accept_multiple_files=True,
                key="clf_uploads",
            )
            for u in ups or []:
                selections.append((u.name, u, "upload", pp_mod.detect_year(u.name)))
        else:
            avail = {}
            for label, target in cfg.QUICK_LOAD_FILES.items():
                if target == "__ALL_RAW__":
                    if all(cfg.resolve_dataset_file(fn).exists() for _, fn in cfg.RAW_YEAR_FILES):
                        avail[label] = target
                elif cfg.resolve_dataset_file(target).exists():
                    avail[label] = target

            choice_label = st.selectbox(
                "Pilih Dataset",
                list(avail.keys()),
                index=0,
                key="clf_quick_pick",
            )
            chosen_target = avail.get(choice_label)
            if chosen_target == "__ALL_RAW__":
                for yr, fn in cfg.RAW_YEAR_FILES:
                    selections.append((fn, fn, "workspace", yr))
            elif chosen_target:
                selections.append((chosen_target, chosen_target, "workspace", pp_mod.detect_year(chosen_target)))

    with st.container(key="card-clf-run"):
        ui.card_header(
            "2",
            "Classification — Pemodelan LightGBM Otomatis",
        )
        run_clicked = st.button(
            "Jalankan Pre-processing & Klasifikasi LightGBM",
            type="primary",
            disabled=not selections,
            width="stretch",
            key="btn_run_pipeline",
        )

        if run_clicked and selections:
            bar = st.progress(0.0)
            txt = st.empty()
            cb = _progress_cb(bar, txt)
            try:
                t0 = time.perf_counter()
                sources = []
                for i, (name, handle, kind, auto_yr) in enumerate(selections):
                    cb(0.02 + 0.18 * i / len(selections), f"Membaca {name}…")
                    df = (
                        pp_mod.load_uploaded(handle, bbox_pushdown=True)
                        if kind == "upload"
                        else pp_mod.load_workspace_file(handle, bbox_pushdown=True)
                    )
                    yr = None if "year" in df.columns else auto_yr
                    sources.append(pp_mod.SourceFrame(name=name, df=df, year=yr))

                pp_res = pp_mod.run_pipeline(
                    sources,
                    pp_mod.PipelineOptions(
                        do_bbox=True,
                        do_sjoin=True,
                        reverify_provinsi=False,
                        run_benchmark=False,
                    ),
                    lambda p, m: cb(0.20 + 0.25 * p, m),
                )

                tc = modelling.TrainConfig()
                train_res = modelling.run_training(
                    pp_res.df,
                    tc,
                    lambda p, m: cb(0.45 + 0.50 * p, m),
                )
                total_elapsed = time.perf_counter() - t0
                pp_res.stats["elapsed_sec"] = total_elapsed
                train_res["total_sec"] = total_elapsed
                rec_res = modelling.recommend(train_res)

                cb(0.96, "Menyusun grafik interaktif dan peta hasil klasifikasi…")

                st.session_state.pp = pp_res
                st.session_state.results = train_res
                st.session_state.rec = rec_res
                st.session_state.charts = {}
                st.session_state.pp_charts = {}
                st.session_state.exports = {}
                st.session_state.dl_cache = {}
                st.session_state.clf_done = True

                bar.empty()
                txt.empty()
                st.toast(f"Klasifikasi selesai dalam {total_elapsed:.1f} detik.")
                st.rerun()
            except Exception as e:
                bar.empty()
                txt.empty()
                st.error(f"Proses gagal: {e}")

        with st.expander("Uji Prediksi Cepat 1 Data Traffic Jaringan (Opsional)", expanded=False):
            _render_single_simulator()


# ══════════════════════════════════════════════════════════════════════════════
# Optional Single-Traffic Quick Simulator
# ══════════════════════════════════════════════════════════════════════════════
def _apply_preset(name: str) -> None:
    for f, v in cfg.PRESETS[name].items():
        st.session_state[f"inp_{f}"] = float(v) if isinstance(v, float) else int(v)


def _render_single_simulator() -> None:
    home.ensure_state_loaded()
    res = st.session_state.get("results")
    if not res or not res.get("splits"):
        st.info("Model belum tersedia.")
        return

    st.caption("Pilih preset profil kualitas jaringan atau ubah nilai metrik di bawah:")
    pc = st.columns(len(cfg.CLASS_NAMES))
    for col, name in zip(pc, cfg.CLASS_NAMES):
        col.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,), width="stretch")
    for f, v in cfg.PRESETS["Good"].items():
        st.session_state.setdefault(f"inp_{f}", v)

    c = st.columns(4)
    vals = {
        "avg_d_kbps": c[0].number_input("Download (kbps)", 0, 5_000_000, step=500, key="inp_avg_d_kbps"),
        "avg_u_kbps": c[1].number_input("Upload (kbps)", 0, 5_000_000, step=500, key="inp_avg_u_kbps"),
        "avg_lat_ms": c[2].number_input("Idle Latency (ms)", 0, 20_000, step=1, key="inp_avg_lat_ms"),
        "avg_lat_down_ms": c[3].number_input("DL Latency (ms)", 0.0, 60_000.0, step=10.0, key="inp_avg_lat_down_ms"),
    }
    c = st.columns(3)
    vals["avg_lat_up_ms"] = c[0].number_input("UL Latency (ms)", 0.0, 60_000.0, step=10.0, key="inp_avg_lat_up_ms")
    vals["tests"] = c[1].slider("Jumlah Tests", 1, 200, key="inp_tests")
    vals["devices"] = c[2].slider("Jumlah Devices", 1, 100, key="inp_devices")

    X = pd.DataFrame([vals])[cfg.FEATURES].astype("float64")
    cols = st.columns(len(res["splits"]))
    for col, (sname, r) in zip(cols, res["splits"].items()):
        mdl = r["model"]
        t0 = time.perf_counter()
        proba = mdl.predict_proba(X)[0]
        ms = (time.perf_counter() - t0) * 1000
        code = int(np.argmax(proba))
        label = cfg.CLASS_NAMES[code]
        conf = float(proba[code]) * 100
        col.html(
            f"""
            <div class="pred-card">
              <div class="pred-head"><b style="color:#c4b5fd">Model LightGBM</b>{ui.split_badge(sname)}</div>
              <div class="pred-main">{ui.cls_pill(label, big=True)}
                <div><div style="font-size:.78rem;color:#a5b4fc">Keyakinan Prediksi</div>
                <div style="font-family:var(--font-yuruka);font-size:1.5rem;font-weight:800">{conf:.2f}%</div></div>
              </div>
              <div class="pred-kv"><span>Waktu Inferensi</span><b>{ms:.2f} ms</b></div>
              <div class="pred-interp"><b>{label}</b>: {html.escape(cfg.QUALITY_INTERPRETATION[label])}</div>
            </div>
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Result (Displayed Directly Below After Modelling Finishes)
# ══════════════════════════════════════════════════════════════════════════════
def _render_results_below() -> None:
    if not st.session_state.get("clf_done"):
        with st.container(key="card-clf-result-pending"):
            ui.card_header(
                "3",
                "Result — Hasil Klasifikasi, Evaluasi & Peta Sebaran",
            )
        return

    pp = st.session_state.get("pp")
    res = st.session_state.get("results")
    rec = st.session_state.get("rec")
    if pp is None or res is None or rec is None:
        return

    s = pp.stats
    sp = res["splits"]
    winner = rec["winner"]
    ch = st.session_state.get("charts") or {}

    # ── 3A. Summary KPIs & Side-by-Side Recommendation Table ──────────────────
    with st.container(key="card-clf-result"):
        ui.card_header(
            "3",
            "Result — Hasil Pre-processing & Pemodelan LightGBM",
        )
        ui.kpis([
            ("Baris Awal", f"{s['initial_rows']:,}", "", ""),
            ("Luar Indonesia / Laut", f"{s['non_indonesia_removed']:,}", "", "bad"),
            ("Missing Value / Unknown", f"{s['missing_removed']:,}", "", "bad"),
            ("Data Bersih Indonesia", f"{s['final_rows']:,}", f"{s['provinces']} Provinsi", "good"),
            *[
                (f"Akurasi Uji ({sname})", f"{r['accuracy'] * 100:.2f}%",
                 f"Macro F1: {r['macro_f1']:.4f}", "good" if sname == winner else "purple")
                for sname, r in sp.items()
            ],
        ])
        ui.render_recommendation(rec, res)

    # ── 3B. Interactive Geospatial Map of the Result ──────────────────────────
    with st.container(key="card-clf-map"):
        ui.card_header(
            None,
            "Peta Hasil Klasifikasi Kualitas Jaringan Indonesia",
        )
        maps.render_map_section(pp.df, key_prefix="clf")

    # ── 3C. Tabs: Grafik | Confusion Matrix | Evaluasi | Export ───────────────
    with st.container(key="card-clf-charts"):
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
                    key="clf_fig_dist",
                )
            with c2:
                if not pp.prov_quality_df.empty:
                    st.plotly_chart(
                        charts.fig_top_provinces(pp.prov_quality_df, 15),
                        width="stretch",
                        config=charts.PLOTLY_CONFIG,
                        key="clf_fig_prov",
                    )
            st.plotly_chart(
                charts.fig_global_comparison(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="clf_fig_global",
            )
            st.plotly_chart(
                charts.fig_per_class(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="clf_fig_per_class",
            )
            st.plotly_chart(
                charts.fig_feature_importance(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="clf_fig_fi",
            )

        with tabs[1]:
            st.plotly_chart(
                charts.fig_confusion_matrices(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="clf_fig_cm",
            )
            for sname, r in sp.items():
                ui.table(r["cm_df"], f"Tabel Confusion Matrix — {sname}", index=True, diag=True)
            st.plotly_chart(
                charts.fig_logloss_curves(res),
                width="stretch",
                config=charts.PLOTLY_CONFIG,
                key="clf_fig_logloss",
            )
            if any(r.get("cv_df") is not None for r in sp.values()):
                st.plotly_chart(
                    charts.fig_kfold_lines(res),
                    width="stretch",
                    config=charts.PLOTLY_CONFIG,
                    key="clf_fig_kfold",
                )

        with tabs[2]:
            ui.table(
                res["comparison_df"],
                "Perbandingan Performa Global & Waktu Komputasi (80:20 vs 70:30)",
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
            cvo = res.get("cv_overview_df")
            if cvo is not None and not cvo.empty:
                ui.table(
                    cvo,
                    f"Ringkasan {res['config']['k_folds']}-Fold Cross Validation",
                    best_cols={"Mean CV Accuracy": "max", "Std CV Accuracy": "min", "Mean Macro F1-Score": "max"},
                )
            ui.table(res["fi_df"], "Feature Importance (Split & Gain)")

        with tabs[3]:
            exporter.render_export_panel(pp, res, rec, ch, key_prefix="clf")


def render() -> None:
    _render_data_and_run()
    _render_results_below()
