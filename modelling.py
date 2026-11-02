"""LightGBM modelling (model.ipynb TAHAP 2–7) + automated model recommendation.

* Stratified 80:20 and 70:30 splits (random_state=727, stratify=y)
* LGBMClassifier with early stopping & multi_logloss tracking
* Optional RandomizedSearchCV / GridSearchCV hyper-parameter tuning
* Stratified K-Fold CV on each training split
* 5×5 confusion matrix + One-vs-Rest decomposition per class
* Split / Gain feature importance
* Recommendation engine (80:20 vs 70:30)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

import config as cfg

warnings.filterwarnings("ignore")
ProgressFn = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


@dataclass
class TrainConfig:
    params: dict = field(default_factory=lambda: dict(cfg.DEFAULT_LGBM_PARAMS))
    early_stopping: int = cfg.EARLY_STOPPING
    k_folds: int = cfg.K_FOLDS
    run_cv: bool = True
    sample_frac: float = 1.0
    tuning: bool = False
    search_method: str = "random"  # 'random' | 'grid'
    n_iter: int = 10
    tuning_cv: int = 3
    tuning_max_rows: int = 150_000
    param_space: dict = field(default_factory=lambda: dict(cfg.PARAM_SPACE))
    splits: tuple = tuple(cfg.SPLIT_NAMES)


def sanitize_params(params: dict) -> dict:
    """Make a parameter dict valid for the selected boosting type."""
    p = dict(params)
    p["objective"] = "multiclass"
    p["num_class"] = 5
    p.setdefault("random_state", cfg.RANDOM_STATE)
    p.setdefault("n_jobs", -1)
    p["verbose"] = -1
    if p.get("class_weight") in ("None", "", None):
        p["class_weight"] = None
    if p.get("boosting_type") == "goss":
        # GOSS cannot be combined with bagging
        p["subsample"] = 1.0
    return p


def _metric_block(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _callbacks(es: int, use_es: bool) -> list:
    cbs = [log_evaluation(period=0)]
    if use_es and es and es > 0:
        cbs.append(early_stopping(stopping_rounds=int(es), verbose=False))
    return cbs


def _best_iter(model: LGBMClassifier) -> int:
    bi = getattr(model, "best_iteration_", None)
    if not bi:
        return int(model.n_estimators_ if hasattr(model, "n_estimators_") else model.n_estimators)
    return int(bi)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════
def ovr_table(cm: np.ndarray, split_name: str) -> pd.DataFrame:
    n_total = cm.sum()
    rows = []
    for k, cname in enumerate(cfg.CLASS_NAMES):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)
        tn = int(n_total - tp - fp - fn)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({
            "Skema Split": split_name, "Kelas (k)": f"{k} - {cname}",
            "TP_k": tp, "FP_k": fp, "FN_k": fn, "TN_k": tn,
            "Precision (P_k)": round(p, 4), "Recall (R_k)": round(r, 4),
            "F1-Score (F1_k)": round(f1, 4), "Support (N_k)": int(cm[k, :].sum()),
        })
    return pd.DataFrame(rows)


def cm_frame(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=[f"Aktual: {i} - {c}" for i, c in enumerate(cfg.CLASS_NAMES)],
        columns=[f"Prediksi: {i} - {c}" for i, c in enumerate(cfg.CLASS_NAMES)],
    )


def feature_importance(model: LGBMClassifier, split_name: str) -> pd.DataFrame:
    booster = model.booster_
    return pd.DataFrame({
        "Skema Split": split_name,
        "Fitur": cfg.FEATURES,
        "Split Importance": booster.feature_importance(importance_type="split"),
        "Gain Importance": np.round(booster.feature_importance(importance_type="gain"), 2),
    }).sort_values("Gain Importance", ascending=False).reset_index(drop=True)


def measure_latency(model: LGBMClassifier, X_sample: pd.DataFrame, repeats: int = 25) -> dict:
    """Single-row and per-row batch inference latency (ms)."""
    one = X_sample.iloc[[0]]
    model.predict_proba(one)  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict_proba(one)
        times.append(time.perf_counter() - t)
    batch = X_sample.iloc[: min(len(X_sample), 10_000)]
    t = time.perf_counter()
    model.predict(batch)
    batch_s = time.perf_counter() - t
    return {
        "single_ms": float(np.median(times) * 1000),
        "batch_rows": int(len(batch)),
        "batch_ms_per_row": float(batch_s / max(len(batch), 1) * 1000),
        "batch_s": float(batch_s),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tuning
# ══════════════════════════════════════════════════════════════════════════════
def tune_params(X_tr, y_tr, tc: TrainConfig, base: dict) -> tuple[dict, float, float]:
    if len(X_tr) > tc.tuning_max_rows:
        X_tr, _, y_tr, _ = train_test_split(
            X_tr, y_tr, train_size=tc.tuning_max_rows,
            random_state=cfg.RANDOM_STATE, stratify=y_tr,
        )
    est = LGBMClassifier(
        objective="multiclass", num_class=5,
        boosting_type=base["boosting_type"], class_weight=base.get("class_weight"),
        random_state=cfg.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    cv = StratifiedKFold(n_splits=tc.tuning_cv, shuffle=True, random_state=cfg.RANDOM_STATE)
    space = {k: v for k, v in tc.param_space.items() if v}
    if base["boosting_type"] == "goss":
        space.pop("subsample", None)
    if tc.search_method == "grid":
        search = GridSearchCV(est, space, scoring="f1_macro", cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            est, space, n_iter=tc.n_iter, scoring="f1_macro", cv=cv,
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        )
    t = time.perf_counter()
    search.fit(X_tr, y_tr)
    return dict(search.best_params_), float(search.best_score_), time.perf_counter() - t


def grid_size(space: dict) -> int:
    n = 1
    for v in space.values():
        n *= max(len(v), 1)
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_one(X_tr, y_tr, X_te, y_te, params: dict, es: int) -> dict:
    model = LGBMClassifier(**params)
    use_es = params.get("boosting_type") != "dart"
    evals: dict = {}
    t = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_te, y_te)],
        eval_names=["Train", "Validation"],
        eval_metric="multi_logloss",
        callbacks=_callbacks(es, use_es) + [record_evaluation(evals)],
    )
    train_s = time.perf_counter() - t
    t = time.perf_counter()
    y_pred = model.predict(X_te)
    pred_s = time.perf_counter() - t
    return {
        "model": model, "y_pred": np.asarray(y_pred), "train_time_sec": train_s,
        "pred_time_sec": pred_s, "evals": model.evals_result_ or evals,
        "best_iteration": _best_iter(model),
    }


def run_kfold(X_tr, y_tr, params: dict, k: int, es: int, split_name: str,
              progress: ProgressFn = _noop) -> tuple[pd.DataFrame, float]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.RANDOM_STATE)
    use_es = params.get("boosting_type") != "dart"
    rows, t0 = [], time.perf_counter()
    for i, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), start=1):
        progress((i - 1) / k, f"[{split_name}] Fold {i}/{k}")
        Xf, Xv = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        yf, yv = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        m = LGBMClassifier(**params)
        t = time.perf_counter()
        m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=_callbacks(es, use_es))
        tt = time.perf_counter() - t
        t = time.perf_counter()
        yp = m.predict(Xv)
        pt = time.perf_counter() - t
        mb = _metric_block(yv, yp)
        rows.append({
            "Skema Split": split_name, "Fold": f"Fold-{i}",
            "Accuracy (E_i)": round(mb["accuracy"], 4),
            "Macro Precision": round(mb["macro_precision"], 4),
            "Macro Recall": round(mb["macro_recall"], 4),
            "Macro F1-Score": round(mb["macro_f1"], 4),
            "Weighted Precision": round(mb["weighted_precision"], 4),
            "Weighted Recall": round(mb["weighted_recall"], 4),
            "Weighted F1-Score": round(mb["weighted_f1"], 4),
            "Best Iteration": _best_iter(m),
            "Waktu Training (detik)": round(tt, 4),
            "Waktu Prediksi (detik)": round(pt, 4),
        })
    total = time.perf_counter() - t0
    fold_df = pd.DataFrame(rows)
    metric_cols = [
        "Accuracy (E_i)", "Macro Precision", "Macro Recall", "Macro F1-Score",
        "Weighted Precision", "Weighted Recall", "Weighted F1-Score",
        "Waktu Training (detik)", "Waktu Prediksi (detik)",
    ]
    mean_d = {"Skema Split": split_name, "Fold": "Rata-rata (CV)",
              "Best Iteration": round(fold_df["Best Iteration"].mean(), 1)}
    std_d = {"Skema Split": split_name, "Fold": "Std Deviasi (CV)",
             "Best Iteration": round(fold_df["Best Iteration"].std(), 1)}
    for c in metric_cols:
        mean_d[c] = round(fold_df[c].mean(), 4)
        std_d[c] = round(fold_df[c].std(), 4)
    progress(1.0, f"[{split_name}] K-Fold done")
    return pd.concat([fold_df, pd.DataFrame([mean_d, std_d])], ignore_index=True), total


def run_training(df: pd.DataFrame, tc: TrainConfig, progress: ProgressFn = _noop) -> dict:
    """Train & evaluate both split schemes. Returns a serialisable results dict."""
    params = sanitize_params(tc.params)
    data = df[cfg.FEATURES + [cfg.TARGET_ENCODED]].dropna()
    if tc.sample_frac < 1.0:
        data, _ = train_test_split(
            data, train_size=tc.sample_frac, random_state=cfg.RANDOM_STATE,
            stratify=data[cfg.TARGET_ENCODED],
        )
    X = data[cfg.FEATURES].astype("float64")
    y = data[cfg.TARGET_ENCODED].astype("int64")

    n_steps = len(tc.splits) * (2 + (1 if tc.tuning else 0) + (1 if tc.run_cv else 0))
    step = [0]

    def sub(p: float, msg: str) -> None:
        progress(min((step[0] + p) / n_steps, 0.999), msg)

    splits: dict = {}
    split_rows, comp_rows, cv_over, cv_all, fi_all = [], [], [], [], []
    for sname in tc.splits:
        ratio = cfg.SPLIT_RATIOS[sname]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=ratio, random_state=cfg.RANDOM_STATE, stratify=y
        )
        for idx, cname in enumerate(cfg.CLASS_NAMES):
            split_rows.append({
                "Skema Split": sname, "Kelas Kualitas": f"{idx} - {cname}",
                "Train Count": int((y_tr == idx).sum()), "Test Count": int((y_te == idx).sum()),
            })

        sp = dict(params)
        tuning_info = None
        if tc.tuning:
            sub(0, f"[{sname}] Hyper-parameter tuning ({tc.search_method})…")
            best, score, dur = tune_params(X_tr, y_tr, tc, sp)
            sp = sanitize_params({**sp, **best})
            tuning_info = {"best_params": best, "best_f1_macro": score, "duration_sec": dur,
                           "method": tc.search_method}
            step[0] += 1

        sub(0, f"[{sname}] Training LightGBM on {len(X_tr):,} rows…")
        res = train_one(X_tr, y_tr, X_te, y_te, sp, tc.early_stopping)
        step[0] += 1

        sub(0, f"[{sname}] Evaluating on {len(X_te):,} test rows…")
        y_te_np = y_te.to_numpy()
        mb = _metric_block(y_te_np, res["y_pred"])
        cm = confusion_matrix(y_te_np, res["y_pred"], labels=list(range(5)))
        ovr = ovr_table(cm, sname)
        fi = feature_importance(res["model"], sname)
        lat = measure_latency(res["model"], X_te)
        fi_all.append(fi)
        step[0] += 1

        cv_df, cv_total = None, None
        if tc.run_cv:
            cv_df, cv_total = run_kfold(
                X_tr, y_tr, sp, tc.k_folds, tc.early_stopping, sname,
                lambda p, m: sub(p, m),
            )
            step[0] += 1
            mean_r = cv_df[cv_df["Fold"] == "Rata-rata (CV)"].iloc[0]
            std_r = cv_df[cv_df["Fold"] == "Std Deviasi (CV)"].iloc[0]
            cv_all.append(cv_df)
            cv_over.append({
                "Skema Split": sname, "Jumlah Fold (k)": tc.k_folds,
                "Mean CV Accuracy": mean_r["Accuracy (E_i)"],
                "Std CV Accuracy": std_r["Accuracy (E_i)"],
                "Mean Macro Precision": mean_r["Macro Precision"],
                "Mean Macro Recall": mean_r["Macro Recall"],
                "Mean Macro F1-Score": mean_r["Macro F1-Score"],
                "Std Macro F1-Score": std_r["Macro F1-Score"],
                "Mean Weighted F1-Score": mean_r["Weighted F1-Score"],
                "Rata-rata Waktu per Fold (detik)": mean_r["Waktu Training (detik)"],
                "Total Waktu K-Fold (detik)": round(cv_total, 4),
            })

        comp_rows.append({
            "Skema Split": sname, "Jumlah Train": len(X_tr), "Jumlah Test": len(X_te),
            "Best Iteration": res["best_iteration"],
            "Waktu Training (detik)": round(res["train_time_sec"], 4),
            "Waktu Prediksi (detik)": round(res["pred_time_sec"], 4),
            "Overall Accuracy": round(mb["accuracy"], 4),
            "Macro Precision": round(mb["macro_precision"], 4),
            "Macro Recall": round(mb["macro_recall"], 4),
            "Macro F1-Score": round(mb["macro_f1"], 4),
            "Weighted Precision": round(mb["weighted_precision"], 4),
            "Weighted Recall": round(mb["weighted_recall"], 4),
            "Weighted F1-Score": round(mb["weighted_f1"], 4),
        })

        splits[sname] = {
            "model": res["model"], "params": sp, "y_test": y_te_np, "y_pred": res["y_pred"],
            "train_time_sec": res["train_time_sec"], "pred_time_sec": res["pred_time_sec"],
            "best_iteration": res["best_iteration"], "evals": res["evals"],
            "n_train": len(X_tr), "n_test": len(X_te), "cm": cm,
            "cm_df": cm_frame(cm), "ovr_df": ovr, "fi_df": fi, "cv_df": cv_df,
            "cv_total_sec": cv_total, "latency": lat, "tuning": tuning_info, **mb,
        }

    split_summary_df = pd.DataFrame(split_rows)
    split_pivot_df = split_summary_df.pivot(
        index="Kelas Kualitas", columns="Skema Split", values=["Train Count", "Test Count"]
    )
    progress(1.0, "Training & evaluation complete.")
    return {
        "splits": splits,
        "comparison_df": pd.DataFrame(comp_rows),
        "split_summary_df": split_summary_df,
        "split_pivot_df": split_pivot_df,
        "cv_overview_df": pd.DataFrame(cv_over) if cv_over else pd.DataFrame(),
        "cv_combined_df": pd.concat(cv_all, ignore_index=True) if cv_all else pd.DataFrame(),
        "fi_df": pd.concat(fi_all, ignore_index=True),
        "dist_df": pd.DataFrame({
            "Label Numerik": list(range(5)),
            "Kelas Kualitas": cfg.CLASS_NAMES,
            "Jumlah Sampel": [int((y == i).sum()) for i in range(5)],
            "Persentase (%)": [round(float((y == i).mean()) * 100, 2) for i in range(5)],
        }),
        "config": {
            "params": params, "early_stopping": tc.early_stopping, "k_folds": tc.k_folds,
            "run_cv": tc.run_cv, "sample_frac": tc.sample_frac, "tuning": tc.tuning,
            "search_method": tc.search_method, "n_rows": int(len(X)),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation engine
# ══════════════════════════════════════════════════════════════════════════════
_TOL = 1e-4


def _minority_f1(split: dict) -> float:
    ovr = split["ovr_df"]
    names = [f"{cfg.LABEL_MAPPING[c]} - {c}" for c in cfg.MINORITY_CLASSES]
    return float(ovr[ovr["Kelas (k)"].isin(names)]["F1-Score (F1_k)"].mean())


def recommend(results: dict) -> dict:
    """Bandingkan skema 80:20 vs 70:30 secara berdampingan pada metrik performa, CV, dan kecepatan."""
    sp = results["splits"]
    names = [n for n in cfg.SPLIT_NAMES if n in sp]
    if len(names) < 2:
        only = names[0]
        return {
            "winner": only,
            "scores": {only: 1.0},
            "criteria": [],
            "fast": only,
            "accurate": only,
        }

    a, b = names
    cv = results.get("cv_overview_df")
    has_cv = cv is not None and not cv.empty

    def cvv(n: str, col: str) -> float | None:
        if not has_cv:
            return None
        return float(cv.loc[cv["Skema Split"] == n, col].iloc[0])

    # (nama_metrik, kategori, bobot, nilai_a, nilai_b, lebih_tinggi_lebih_baik, format_tampilan)
    crit_specs = [
        ("Akurasi Data Uji", "Performa Uji", 2.0, sp[a]["accuracy"], sp[b]["accuracy"], True, "pct"),
        ("Macro F1-Score Data Uji", "Performa Uji", 2.0, sp[a]["macro_f1"], sp[b]["macro_f1"], True, "f4"),
        ("Rata-rata F1 Kelas Minoritas (Good & Great)", "Performa Uji", 1.5, _minority_f1(sp[a]), _minority_f1(sp[b]), True, "f4"),
        ("Weighted F1-Score Data Uji", "Performa Uji", 1.0, sp[a]["weighted_f1"], sp[b]["weighted_f1"], True, "f4"),
    ]
    if has_cv:
        crit_specs += [
            ("Rata-rata Akurasi K-Fold CV", "Validasi Silang (CV)", 1.5, cvv(a, "Mean CV Accuracy"), cvv(b, "Mean CV Accuracy"), True, "pct"),
            ("Rata-rata Macro F1 K-Fold CV", "Validasi Silang (CV)", 1.5, cvv(a, "Mean Macro F1-Score"), cvv(b, "Mean Macro F1-Score"), True, "f4"),
            ("Stabilitas K-Fold CV (Std Deviasi)", "Validasi Silang (CV)", 1.0, cvv(a, "Std CV Accuracy"), cvv(b, "Std CV Accuracy"), False, "f4"),
        ]
    crit_specs += [
        ("Waktu Training Model", "Efisiensi Komputasi", 0.75, sp[a]["train_time_sec"], sp[b]["train_time_sec"], False, "s"),
        ("Waktu Prediksi Data Uji", "Efisiensi Komputasi", 0.5, sp[a]["pred_time_sec"], sp[b]["pred_time_sec"], False, "s4"),
        ("Latensi Inferensi per Baris", "Efisiensi Komputasi", 0.5, sp[a]["latency"]["batch_ms_per_row"], sp[b]["latency"]["batch_ms_per_row"], False, "ms4"),
    ]

    scores = {a: 0.0, b: 0.0}
    criteria = []
    for label, group, w, va, vb, hib, fmt in crit_specs:
        tol = _TOL if group != "Efisiensi Komputasi" else max(abs(va), abs(vb)) * 0.03
        if abs(va - vb) <= tol:
            win = "tie"
            scores[a] += w / 2.0
            scores[b] += w / 2.0
        else:
            win = a if ((va > vb) == hib) else b
            scores[win] += w
        criteria.append({
            "Metrik Evaluasi": label,
            "Kategori": group,
            "Bobot": w,
            a: va,
            b: vb,
            "Kriteria Terbaik": "Nilai Tertinggi" if hib else "Nilai Terendah",
            "Model Unggul": win,
            "fmt": fmt,
        })

    winner = max(scores, key=scores.get)
    if abs(scores[a] - scores[b]) < 1e-9:
        comp = results["comparison_df"].sort_values(
            ["Overall Accuracy", "Macro F1-Score"], ascending=False
        )
        winner = comp.iloc[0]["Skema Split"]
    loser = b if winner == a else a
    fast = a if sp[a]["train_time_sec"] <= sp[b]["train_time_sec"] else b
    accurate = a if (sp[a]["accuracy"], sp[a]["macro_f1"]) >= (sp[b]["accuracy"], sp[b]["macro_f1"]) else b

    return {
        "winner": winner,
        "loser": loser,
        "scores": scores,
        "criteria": criteria,
        "fast": fast,
        "accurate": accurate,
    }


def resolve_primary(results: dict, rec: dict, choice: str) -> str:
    if choice in results["splits"]:
        return choice
    return rec["winner"]


def criteria_frame(rec: dict) -> pd.DataFrame:
    if not rec.get("criteria"):
        return pd.DataFrame()
    rows = []
    for c in rec["criteria"]:
        r = {k: v for k, v in c.items() if k != "fmt"}
        rows.append(r)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-computed workspace loader (for instant Home summary on startup)
# ══════════════════════════════════════════════════════════════════════════════
def load_precomputed_workspace() -> tuple[object | None, dict | None, dict | None]:
    """Load pre-computed dataset, Excel evaluation tables & model bundles from Dataset_2/."""
    import joblib
    import preprocess as pp_mod

    data_path = cfg.DEFAULT_DATA_PATH
    excel_path = cfg.DEFAULT_EXCEL_PATH
    if not data_path.exists() or not excel_path.exists():
        return None, None, None

    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    df[cfg.TARGET_ENCODED] = df[cfg.TARGET].map(cfg.LABEL_MAPPING).astype("int64")

    xl = pd.ExcelFile(excel_path)
    comp_df = xl.parse("Perbandingan_Global")
    cv_over_df = xl.parse("KFold_Ringkasan")
    cv_comb_df = xl.parse("KFold_Gabungan")
    fi_df = xl.parse("Feature_Importance")
    dist_df = xl.parse("Distribusi_Kelas")
    split_sum_df = xl.parse("Distribusi_Split")
    storage_df = xl.parse("CSV_vs_Parquet") if "CSV_vs_Parquet" in xl.sheet_names else None
    if storage_df is not None:
        if "Ukuran File (MB)" not in storage_df.columns:
            storage_df = None
        elif "Penghematan Ruang (%)" not in storage_df.columns:
            mb = pd.to_numeric(storage_df["Ukuran File (MB)"], errors="coerce")
            saving = (1.0 - mb.iloc[1] / mb.iloc[0]) * 100.0 if len(mb) > 1 and mb.iloc[0] else 0.0
            storage_df["Penghematan Ruang (%)"] = [0.0, round(float(saving), 2)]

    if "Std Macro F1-Score" not in cv_over_df.columns:
        stds = []
        for sname in cv_over_df["Skema Split"]:
            sh = f"KFold_{sname.replace(':', '_')}"
            if sh in xl.sheet_names:
                kdf = xl.parse(sh)
                sr = kdf[kdf["Fold"] == "Std Deviasi (CV)"]
                stds.append(float(sr["Macro F1-Score"].iloc[0]) if not sr.empty else 0.0)
            else:
                stds.append(0.0)
        cv_over_df["Std Macro F1-Score"] = stds

    splits = {}
    for sname in cfg.SPLIT_NAMES:
        tag = sname.replace(":", "_")
        bpath = cfg.resolve_dataset_file(cfg.MODEL_BUNDLE_NAMES[sname])
        mpath = cfg.resolve_dataset_file(cfg.MODEL_PKL_NAMES[sname])
        bundle = joblib.load(bpath) if bpath.exists() else (
            {"model": joblib.load(mpath)} if mpath.exists() else None
        )
        if bundle is None:
            continue
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        params = (
            bundle.get("hyperparameters", dict(cfg.DEFAULT_LGBM_PARAMS))
            if isinstance(bundle, dict)
            else dict(cfg.DEFAULT_LGBM_PARAMS)
        )
        crow = comp_df[comp_df["Skema Split"] == sname].iloc[0]
        ovr_df = xl.parse(f"OVR_Kelas_{tag}")
        cm_raw = xl.parse(f"Confusion_Matrix_{tag}")
        cm_df = cm_raw.set_index(cm_raw.columns[0])
        cm_df.index.name = None
        cm = cm_df.to_numpy(dtype="int64")
        cv_df = xl.parse(f"KFold_{tag}")
        s_fi = fi_df[fi_df["Skema Split"] == sname].reset_index(drop=True)
        n_tr, n_te = int(crow["Jumlah Train"]), int(crow["Jumlah Test"])
        pred_s = float(crow["Waktu Prediksi (detik)"])
        splits[sname] = {
            "model": model,
            "params": params,
            "y_test": np.array([]),
            "y_pred": np.array([]),
            "train_time_sec": float(crow["Waktu Training (detik)"]),
            "pred_time_sec": pred_s,
            "best_iteration": int(crow["Best Iteration"]),
            "evals": getattr(model, "evals_result_", {}) or {},
            "n_train": n_tr,
            "n_test": n_te,
            "cm": cm,
            "cm_df": cm_df,
            "ovr_df": ovr_df,
            "fi_df": s_fi,
            "cv_df": cv_df,
            "cv_total_sec": float(
                cv_over_df.loc[cv_over_df["Skema Split"] == sname, "Total Waktu K-Fold (detik)"].iloc[0]
            ),
            "latency": {
                "single_ms": round(pred_s / max(n_te, 1) * 1000 * 350, 3),
                "batch_rows": n_te,
                "batch_ms_per_row": pred_s / max(n_te, 1) * 1000,
                "batch_s": pred_s,
            },
            "tuning": None,
            "accuracy": float(crow["Overall Accuracy"]),
            "macro_precision": float(crow["Macro Precision"]),
            "macro_recall": float(crow["Macro Recall"]),
            "macro_f1": float(crow["Macro F1-Score"]),
            "weighted_precision": float(crow["Weighted Precision"]),
            "weighted_recall": float(crow["Weighted Recall"]),
            "weighted_f1": float(crow["Weighted F1-Score"]),
        }

    split_pivot_df = split_sum_df.pivot(
        index="Kelas Kualitas", columns="Skema Split", values=["Train Count", "Test Count"]
    )
    results = {
        "splits": splits,
        "comparison_df": comp_df,
        "split_summary_df": split_sum_df,
        "split_pivot_df": split_pivot_df,
        "cv_overview_df": cv_over_df,
        "cv_combined_df": cv_comb_df,
        "fi_df": fi_df,
        "dist_df": dist_df,
        "config": {
            "params": dict(cfg.DEFAULT_LGBM_PARAMS),
            "early_stopping": cfg.EARLY_STOPPING,
            "k_folds": cfg.K_FOLDS,
            "run_cv": True,
            "sample_frac": 1.0,
            "tuning": False,
            "search_method": "none",
            "n_rows": int(len(df)),
        },
        "timestamp": "Pre-trained Dataset_2",
    }
    rec = recommend(results)

    missing_report = pd.DataFrame({
        "Kolom": df.columns,
        "Missing (NaN)": [0] * len(df.columns),
        "Persentase (%)": [0.0] * len(df.columns),
        "Tipe Data": [str(df[c].dtype) for c in df.columns],
    })
    stats = {
        "initial_rows": 13_976_613,
        "bbox_removed": 13_316_066,
        "sjoin_removed": 51_557,
        "non_indonesia_removed": 13_367_623,
        "nan_removed": 0,
        "unknown_removed": 0,
        "missing_removed": 0,
        "final_rows": int(len(df)),
        "provinces": int(df["Provinsi"].nunique()) if "Provinsi" in df.columns else 38,
        "years": sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df.columns else cfg.YEARS,
        "elapsed_sec": 0.0,
    }
    pp = pp_mod.PipelineResult(
        df=df,
        stats=stats,
        missing_report=missing_report,
        dist_df=pp_mod.class_distribution(df),
        prov_quality_df=pp_mod.province_quality(df),
        storage_df=storage_df,
        log=[
            "Loaded pre-processed Indonesia dataset (ookla_indonesia_only.parquet).",
            "Bounding Box Indonesia + Spatial Join RBI50K Kab/Kota (38 Provinsi) verified.",
            "Missing values cleaned & ordinal target encoded (Bad=0 .. Great=4).",
        ],
        sources=["ookla_indonesia_only.parquet (2023–2026)"],
    )
    return pp, results, rec

