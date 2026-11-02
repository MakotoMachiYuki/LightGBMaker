"""About view — thesis documentation & methodology (styled like website/documentation.py)."""

from __future__ import annotations

import streamlit as st

import config as cfg


def _flow(steps: list[str]) -> str:
    parts = []
    for i, s in enumerate(steps):
        cls = "s" if i == 0 else ("e" if i == len(steps) - 1 else "")
        parts.append(f'<span class="fl {cls}">{s}</span>')
    return '<div class="doc-flow">' + '<span class="ar">→</span>'.join(parts) + "</div>"


def _head() -> str:
    return f"""
    <div class="doc">
      <h1>About LightGBMaker</h1>
      <p class="doc-subtitle"><b>{cfg.THESIS_TITLE}</b></p>
      <p class="doc-dataset">
        Mendukung <b>SDG Poin 9: Industri, Inovasi, dan Infrastruktur</b> · Dataset:
        <a href="https://registry.opendata.aws/speedtest-global-performance/" target="_blank" rel="noopener noreferrer">
        <b>Speedtest&reg; by Ookla&reg; Global Fixed and Mobile Network Performance Maps</b></a>
        (AWS S3 Open Data, mobile, 2023 Q1 – 2026 Q1).
      </p>
    </div>
    """


def _left_html() -> str:
    q = cfg.QUALITY_COLORS
    classes = "".join(
        f'<div class="doc-class-item"><span class="doc-cls-pill" style="--c:{q[c]}">'
        f'{cfg.LABEL_MAPPING[c]} · {c}</span><span>{cfg.QUALITY_INTERPRETATION[c]}</span></div>'
        for c in cfg.CLASS_NAMES
    )
    return f"""
    <section class="doc-card">
      <h3>Dataset Ookla Open Data</h3>
      <p>Data mentah berformat <i>Apache Parquet</i> diunduh dari repositori publik AWS S3
      (<code>s3://ookla-open-data/parquet/performance/type=mobile/</code>). Setiap baris adalah
      satu <i>tile</i> Web-Mercator zoom-16 (±610 m) dengan agregasi kecepatan, latensi, jumlah
      pengujian dan perangkat, serta label kualitas.</p>

      <h3>Pipeline Spasial Indonesia</h3>
      <ul class="doc-list">
        <li><b>Step 1 — Bounding box</b> (<code>EPSG:4326</code>): lon
        <code>[{cfg.MIN_LON}, {cfg.MAX_LON}]</code>, lat <code>[{cfg.MIN_LAT}, {cfg.MAX_LAT}]</code>.</li>
        <li><b>Step 2 — Spatial join poligon</b> terhadap Geodatabase Peta Rupabumi Indonesia
        <code>RBI50K_ADMINISTRASI_KABKOTA_20230907.gdb</code> (layer <code>{cfg.GDB_LAYER}</code>,
        predikat <code>intersects</code>) untuk membuang titik di laut lepas / negara tetangga dan
        memetakan <code>WADMPR</code> → <code>Provinsi</code> (38 provinsi).</li>
        <li><b>Cleaning</b>: hapus kolom <code>quadkey</code>, <code>tile</code>, <code>geometry</code>,
        <code>index_right</code>; hapus baris <code>NaN</code> dan label <code>Unknown</code>.</li>
        <li><b>Encoding ordinal</b>: <code>Bad=0, Serviceable=1, Average=2, Good=3, Great=4</code>.</li>
      </ul>

      <h3>7 Fitur Prediktor</h3>
      <ul class="doc-list">
        <li><span class="doc-tag dl">avg_d_kbps</span> Rata-rata kecepatan unduh (kbps)</li>
        <li><span class="doc-tag ul">avg_u_kbps</span> Rata-rata kecepatan unggah (kbps)</li>
        <li><span class="doc-tag pg">avg_lat_ms</span> Latensi rata-rata kondisi <i>idle</i> (ms)</li>
        <li><span class="doc-tag dl">avg_lat_down_ms</span> Latensi saat dibebani unduhan (ms)</li>
        <li><span class="doc-tag ul">avg_lat_up_ms</span> Latensi saat dibebani unggahan (ms)</li>
        <li><span class="doc-tag">tests</span> &amp; <span class="doc-tag">devices</span>
        Jumlah pengujian &amp; perangkat unik per tile</li>
      </ul>

      <h3>5 Kelas Target Kualitas</h3>
      <div class="doc-classes">{classes}</div>

      <h3>Alur Sistem</h3>
      <h4>Pre-processing</h4>
      {_flow(["Raw Parquet", "BBox Indonesia", "Spatial Join GDB", "Drop NaN/Unknown", "Encode 0–4", "Parquet bersih"])}
      <h4>Training</h4>
      {_flow(["Dataset bersih", "Stratified split 80:20 / 70:30", "LGBMClassifier", "Early stopping", "Evaluasi uji"])}
      <h4>K-Fold Cross Validation</h4>
      {_flow(["Data latih", "StratifiedKFold (k)", "Latih fold i", "Eᵢ, F1ᵢ", "CV = mean ± std"])}
      <h4>Klasifikasi</h4>
      {_flow(["Input 7 fitur", "predict_proba", "argmax", "Kelas + confidence"])}
    </section>
    """


def _credits_html() -> str:
    rows = [
        ("Dataset", "Speedtest&reg; by Ookla&reg; Open Data (AWS S3)"),
        ("Batas Administrasi", "RBI50K Kab/Kota — Peta Rupabumi Indonesia (BIG)"),
        ("Web Framework", "Streamlit (Python SPA)"),
        ("Machine Learning", "LightGBM · Scikit-Learn · Pandas · NumPy · PyArrow"),
        ("Geospatial", "GeoPandas · Shapely · Pyogrio"),
        ("Visualisasi &amp; Laporan", "Matplotlib · Seaborn · OpenPyXL · ReportLab"),
        ("Typography", "FOT-Yuruka Std · Noto Sans"),
    ]
    body = "".join(f'<div class="doc-credit-row"><span>{a}</span><b>{b}</b></div>' for a, b in rows)
    return f'<section class="doc-card"><h3>Teknologi &amp; Credits</h3><div class="doc-credits">{body}</div></section>'


def render() -> None:
    with st.container(key="card-about"):
        st.html(_head())
        left, right = st.columns(2, gap="medium")
        with left:
            st.html(_left_html())
        with right:
            with st.container(key="card-about-math"):
                st.html(
                    '<div class="doc"><h3 style="margin:0">Metodologi LightGBM</h3></div>'
                    '<p class="doc-subtitle" style="font-size:.9rem">LightGBM adalah algoritma '
                    "<i>Gradient Boosting Decision Tree</i> yang membangun pohon secara "
                    "<b>leaf-wise (best-first)</b>: pada setiap iterasi daun dengan reduksi loss terbesar "
                    "yang dipecah. Model aditif setelah iterasi ke-t:</p>"
                )
                st.latex(r"\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta\, f_t(x_i)")
                st.markdown("Objektif diaproksimasi dengan **ekspansi Taylor orde kedua** menggunakan "
                            "gradien $g_i$ dan Hessian $h_i$:")
                st.latex(r"\mathcal{L}^{(t)} \approx \sum_{i=1}^{n}\Big[g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)\Big] + \Omega(f_t),"
                         r"\quad g_i=\frac{\partial \ell}{\partial \hat{y}_i^{(t-1)}},\; h_i=\frac{\partial^2 \ell}{\partial (\hat{y}_i^{(t-1)})^2}")
                st.markdown("Bobot daun optimal dan **gain** pemecahan (dengan regularisasi $\\lambda$):")
                st.latex(r"w_j^{*} = -\frac{G_j}{H_j+\lambda},\qquad "
                         r"\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma")
                st.markdown(
                    "- **GOSS** (*Gradient-based One-Side Sampling*): mempertahankan $a\\%$ instans "
                    "bergradien terbesar dan mengambil sampel acak $b\\%$ dari sisanya (dibobot $\\frac{1-a}{b}$).\n"
                    "- **EFB** (*Exclusive Feature Bundling*): menggabungkan fitur yang saling eksklusif "
                    "menjadi satu bundel untuk mengurangi dimensi histogram.\n"
                    "- **Multiclass**: objektif `multiclass` (softmax) dengan `multi_logloss`, "
                    "$K=5$ kelas."
                )
                st.html('<div class="doc"><h3 style="margin:.6rem 0 0">Confusion Matrix 5×5 (One-vs-Rest)</h3></div>')
                st.markdown("Untuk setiap kelas $k$ dari matriks $C$ berukuran $5\\times5$ dengan total $N$:")
                st.latex(r"TP_k=C_{kk},\quad FP_k=\sum_{i\neq k}C_{ik},\quad FN_k=\sum_{j\neq k}C_{kj},\quad TN_k=N-TP_k-FP_k-FN_k")
                st.latex(r"P_k=\frac{TP_k}{TP_k+FP_k},\quad R_k=\frac{TP_k}{TP_k+FN_k},\quad F1_k=\frac{2P_kR_k}{P_k+R_k}")
                st.latex(r"\text{Accuracy}=\frac{\sum_{k}TP_k}{N},\quad "
                         r"\text{Macro-}F1=\frac{1}{K}\sum_{k=1}^{K}F1_k,\quad "
                         r"\text{Weighted-}F1=\sum_{k=1}^{K}\frac{N_k}{N}F1_k")
                st.html('<div class="doc"><h3 style="margin:.6rem 0 0">K-Fold Cross Validation</h3></div>')
                st.markdown("Data latih dibagi menjadi $k$ fold berstrata; tiap fold bergantian menjadi "
                            "data validasi dengan error/akurasi $E_i$:")
                st.latex(r"CV=\frac{1}{k}\sum_{i=1}^{k}E_i,\qquad \sigma_{CV}=\sqrt{\frac{1}{k-1}\sum_{i=1}^{k}\left(E_i-CV\right)^2}")
                st.markdown(
                    f"Konfigurasi default: `random_state={cfg.RANDOM_STATE}`, `K_FOLDS={cfg.K_FOLDS}`, "
                    f"`early_stopping={cfg.EARLY_STOPPING}`, `n_estimators=500`, `learning_rate=0.05`, "
                    "`num_leaves=31`. Skema **80:20** dan **70:30** dibandingkan berdampingan lalu "
                    "direkomendasikan secara otomatis berdasarkan performa, stabilitas, dan kecepatan."
                )
            st.html(_credits_html())
        st.html(f'<p class="doc-foot">{cfg.APP_NAME} · About &amp; Methodology</p>')
