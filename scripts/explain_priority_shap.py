#!/usr/bin/env python
from __future__ import annotations

"""
Explain & rank retrofit priority with SHAP.

- Loads the trained regression pipeline (and optional classifier pipeline)
  from artifacts/, supporting filenames with or without 'full_' in them.
- Rebuilds feature engineering needed by the model so the CSV doesn't need
  to already contain engineered columns.
- Computes a retrofit priority score that blends:
    * normalized "worse is higher" energy-star risk, and
    * probability of low grade (C/D) if a classifier exists.
  Weight controlled by --alpha (default 0.35).
- Produces:
    artifacts/shap/priority_ranked.csv             (all rows)
    artifacts/shap/priority_topk.csv               (top-k subset)
    artifacts/shap/shap_values_topk.csv            (SHAP on top-k sample)
    artifacts/shap/shap_summary_topk.png           (summary plot)
"""

import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import joblib

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False

# ---------------------- Compatibility helpers for unpickling ------------------

# If your training pipelines were saved with these utilities inside, joblib
# needs to find the same symbols at import time. We provide no-op compatible
# versions here so unpickling works even when this script runs standalone.

def to_float_ndarray(X):
    """Coerce arbitrary DataFrame/ndarray to float64 with np.nan (no pd.NA)."""
    Xdf = pd.DataFrame(X)
    Xnum = Xdf.apply(pd.to_numeric, errors="coerce")
    return Xnum.to_numpy(dtype="float64", copy=False)

def to_object_with_nan(X):
    """Ensure object-dtype array with np.nan for missings (no pandas.NA/NaT)."""
    Xdf = pd.DataFrame(X).astype("object").copy()
    Xdf = Xdf.replace({pd.NA: np.nan})
    try:
        from pandas import NaT
        Xdf = Xdf.replace({NaT: np.nan})
    except Exception:
        pass
    return Xdf.astype(object).to_numpy(dtype=object, copy=False)

# ------------------------------ Feature engineering --------------------------

NUMERIC_BASE = [
    "floor_area", "num_floors", "year_built", "assessed_value"
]
CATEGORICAL_BASE = [
    "borough", "zipcode", "land_use"  # raw source features (used to derive others)
]

ENGINEERED_NUMERIC = [
    "age", "age2", "log_floor_area", "log_assessed_value",
    "value_per_sf",
    "zip_median_year_built", "zip_mean_floor_area",
    "elec_kwh_per_sf", "gas_thm_per_sf"
]
ENGINEERED_CATEGORICAL = [
    "land_use_bucket", "lat_bin", "lon_bin"
]

# Engineer the same features used during training.
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age + nonlinearity
    if {"reporting_year", "year_built"}.issubset(df.columns):
        df["age"] = (df["reporting_year"] - df["year_built"]).clip(lower=0)
        df["age2"] = df["age"] ** 2

    # Logs
    if "floor_area" in df.columns:
        df["log_floor_area"] = np.log1p(df["floor_area"].clip(lower=0))
    if "assessed_value" in df.columns:
        df["log_assessed_value"] = np.log1p(df["assessed_value"].clip(lower=0))

    # Value density
    if {"assessed_value", "floor_area"}.issubset(df.columns):
        denom = df["floor_area"].replace(0, np.nan)
        df["value_per_sf"] = (df["assessed_value"] / denom).replace([np.inf, -np.inf], np.nan)

    # Utilities per SF if present
    if {"elec_kwh", "floor_area"}.issubset(df.columns):
        df["elec_kwh_per_sf"] = (df["elec_kwh"] / df["floor_area"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if {"gas_therms", "floor_area"}.issubset(df.columns):
        df["gas_thm_per_sf"] = (df["gas_therms"] / df["floor_area"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Coarse lat/lon bins
    if {"latitude", "longitude"}.issubset(df.columns):
        # Using Int64 (nullable) so missing stays NA not cast to float
        df["lat_bin"] = (df["latitude"] * 100).round().astype("Int64")
        df["lon_bin"] = (df["longitude"] * 100).round().astype("Int64")

    # Land use buckets from PLUTO landuse codes 1..11
    if "land_use" in df.columns:
        lu = df["land_use"].astype(str).str.strip()
        map_land = {str(i): str(i) for i in range(1, 12)}
        map_land.update({f"{i:02d}": str(i) for i in range(1, 12)})
        lu_code = lu.map(map_land).fillna(lu)
        buckets = {
            "1": "1_res", "2": "2_multi", "3": "3_mixed", "4": "4_commercial",
            "5": "5_industrial", "6": "6_transport", "7": "7_utility",
            "8": "8_public", "9": "9_open", "10": "10_parking", "11": "11_condo"
        }
        df["land_use_bucket"] = lu_code.map(buckets).fillna("other")

    # ZIP aggregates
    if "zipcode" in df.columns:
        z = df["zipcode"].astype(str)
        if "year_built" in df.columns:
            df["zip_median_year_built"] = z.map(df.groupby(z, dropna=False)["year_built"].median())
        if "floor_area" in df.columns:
            df["zip_mean_floor_area"] = z.map(df.groupby(z, dropna=False)["floor_area"].mean())

    # Clip extreme outliers (helps stability)
    for c, hi in [("floor_area", 2_000_000), ("num_floors", 80), ("assessed_value", 5e9)]:
        if c in df.columns:
            df.loc[df[c] > hi, c] = hi

    # Clean impossible values
    for c in ["floor_area", "num_floors", "assessed_value", "year_built", "age", "age2",
              "elec_kwh_per_sf", "gas_thm_per_sf", "value_per_sf"]:
        if c in df.columns:
            df.loc[~np.isfinite(df[c]), c] = np.nan

    return df

# ------------------------------- Model loading -------------------------------

def pick_models(artifacts_dir: pathlib.Path, prefer: str | None):
    """
    Try to load models in this order:
    1) If prefer is 'xgb' or 'rf' or 'full', try that family first (accept *_full_* names).
    2) Else read metadata.json['model_family'] if present and try that family.
    3) Else best-effort: pick any energy_star_regressor_* (prefer xgb over rf).
    Returns (reg_pipeline, clf_pipeline_or_None, chosen_family_or_None)
    """
    import json

    def try_family(fam: str):
        # accept both "..._regressor_xxx.joblib" and "..._regressor_full_xxx.joblib"
        reg = None
        for pat in [f"energy_star_regressor_{fam}.joblib",
                    f"energy_star_regressor_full_{fam}.joblib"]:
            p = artifacts_dir / pat
            if p.exists():
                reg = joblib.load(p)
                break

        clf = None
        for pat in [f"grade_classifier_{fam}.joblib",
                    f"grade_classifier_full_{fam}.joblib"]:
            q = artifacts_dir / pat
            if q.exists():
                try:
                    clf = joblib.load(q)
                except Exception as e:
                    print(f"[warn] Could not load classifier {q.name}: {e}")
                break
        return reg, clf

    families: list[str] = []
    if prefer in ("xgb", "rf", "full_xgb", "full_rf", "full"):
        # 'full' just means "use whatever is there, but try xgb first"
        if prefer == "rf": families = ["rf"]
        elif prefer == "xgb": families = ["xgb"]
        else: families = ["xgb", "rf"]
    else:
        meta_p = artifacts_dir / "metadata.json"
        if meta_p.exists():
            try:
                fam = json.loads(meta_p.read_text()).get("model_family")
                if fam in ("xgb", "rf"):
                    families = [fam]
            except Exception:
                pass
        if not families:
            families = ["xgb", "rf"]

    for fam in families:
        reg, clf = try_family(fam)
        if reg is not None:
            print(f"[info] Using family={fam} reg={reg.__class__.__name__} clf={'none' if clf is None else clf.__class__.__name__}")
            return reg, clf, ("xgb" if "xgb" in fam else "rf")

    # best-effort fallback: any regressor, prefer xgb
    cands = sorted(artifacts_dir.glob("energy_star_regressor*.joblib"))
    if cands:
        xgb = [p for p in cands if "xgb" in p.name.lower()]
        chosen = xgb[0] if xgb else cands[0]
        fam = "xgb" if "xgb" in chosen.name.lower() else "rf"
        reg = joblib.load(chosen)
        clf = None
        for pat in [f"grade_classifier_{fam}.joblib", f"grade_classifier_full_{fam}.joblib"]:
            h = artifacts_dir / pat
            if h.exists():
                try:
                    clf = joblib.load(h)
                except Exception as e:
                    print(f"[warn] Could not load classifier {h.name}: {e}")
                break
        print(f"[info] Fallback using {chosen.name} (family={fam}), classifier={'present' if clf else 'absent'}")
        return reg, clf, fam

    print("No compatible models found in", artifacts_dir)
    return None, None, None

# Extract the original column list the ColumnTransformer was expecting
def expected_input_columns_from_pre(pre) -> list[str]:
    cols = []
    if hasattr(pre, "transformers_"):
        for (_name, _trans, sel) in pre.transformers_:
            if sel is None or sel == "drop":
                continue
            if isinstance(sel, (list, tuple, np.ndarray, pd.Index)):
                cols.extend(list(sel))
            elif isinstance(sel, str):
                cols.append(sel)
    # Deduplicate but preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# --------------------------------- Scoring -----------------------------------

def normalize_inverse(series: pd.Series) -> pd.Series:
    """Normalize so higher = worse, based on 5th-95th percentile band."""
    s = series.astype(float)
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    # worse if lower ENERGY STAR score (closer to 0). So invert: (hi - s) / (hi - lo)
    denom = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    norm = (hi - s).clip(lower=0, upper=(hi - lo)) / denom
    return pd.Series(norm, index=series.index)

def retrofit_priority(y_reg_pred: pd.Series,
                      low_grade_prob: pd.Series | None,
                      alpha: float) -> pd.Series:
    """
    Priority = alpha * (inverse-normalized energy_star) + (1-alpha) * P(low_grade)
    If classifier not available, falls back to inverse-normalized energy_star.
    """
    inv_es = normalize_inverse(y_reg_pred)
    if low_grade_prob is None:
        return inv_es
    lg = low_grade_prob.fillna(0.0).clip(0, 1)
    return alpha * inv_es + (1 - alpha) * lg

# ----------------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Joined/processed CSV (with PLUTO merge)")
    ap.add_argument("--artifacts", default="artifacts", help="Dir with trained pipelines")
    ap.add_argument("--outdir", default="artifacts/shap", help="Where to store outputs")
    ap.add_argument("--alpha", type=float, default=0.35, help="Weight for inverse-normalized energy star in priority")
    ap.add_argument("--topk", type=int, default=2000, help="Top-k rows for SHAP explain")
    ap.add_argument("--prefer", default=None, help="Prefer model family: xgb|rf|full")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load models
    reg, clf, family = pick_models(pathlib.Path(args.artifacts), args.prefer)
    if reg is None:
        print("No compatible models found in artifacts/. Train first.")
        return

    # Identify preprocessor & model objects inside pipeline
    try:
        pre = reg.named_steps.get("pre", None)
        reg_est = reg.named_steps.get("model", reg)
    except Exception:
        pre = None
        reg_est = reg

    clf_est = None
    if clf is not None:
        try:
            clf_est = clf.named_steps.get("model", clf)
        except Exception:
            clf_est = clf

    # Load and engineer features
    df = pd.read_csv(args.csv)
    # Keep common ids if present
    id_cols = [c for c in ["bbl", "address", "zipcode", "borough", "latitude", "longitude"] if c in df.columns]
    df = engineer_features(df)

    # Figure the expected raw input columns for the preprocessor
    if pre is not None:
        expected_cols = expected_input_columns_from_pre(pre)
    else:
        # Fall back to a broad guess: base + engineered
        expected_cols = list(dict.fromkeys(NUMERIC_BASE + CATEGORICAL_BASE + ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL))

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[warn] Missing features in CSV (engineered or source): {missing}")

    # Use intersection: anything we *do* have (preprocessor will impute/ignore as needed)
    feature_cols = [c for c in expected_cols if c in df.columns]
    if not feature_cols:
        raise SystemExit("No usable input features found after intersection with expected preprocessor columns.")

    # Build prediction matrix
    if pre is not None:
        X = df[feature_cols]
        y_star_pred = pd.Series(reg.predict(X), index=df.index, name="energy_star_pred")
    else:
        # If reg is a raw estimator (unlikely), try passing the engineered df directly
        X = df[feature_cols]
        y_star_pred = pd.Series(reg.predict(X), index=df.index, name="energy_star_pred")

    # Low-grade probability if classifier exists (P(C/D))
    low_grade_prob = None
    if clf is not None:
        try:
            # Classifier was trained on labels 0:A,1:B,2:C,3:D for xgb; or strings A-D for RF baseline.
            # We will try both. First, predict_proba; then pick columns for C/D.
            proba = clf.predict_proba(df[feature_cols])
            # Heuristic: if proba has 4 columns, assume order A,B,C,D -> low grade = C or D
            if proba.shape[1] == 4:
                low_grade_prob = pd.Series(proba[:, 2] + proba[:, 3], index=df.index, name="p_low_grade")
            else:
                # If different number, try to use classes_ mapping
                classes = getattr(clf, "classes_", None)
                if classes is not None and len(classes) >= 2:
                    # Try to detect labels for 'C' and 'D'
                    idx_c = list(classes).index("C") if "C" in classes else None
                    idx_d = list(classes).index("D") if "D" in classes else None
                    p = np.zeros(len(df))
                    if idx_c is not None:
                        p += proba[:, idx_c]
                    if idx_d is not None:
                        p += proba[:, idx_d]
                    low_grade_prob = pd.Series(p, index=df.index, name="p_low_grade")
        except Exception as e:
            print(f"[warn] Could not compute classifier probabilities: {e}")

    # Compute priority
    priority = retrofit_priority(y_star_pred, low_grade_prob, alpha=args.alpha)

    # Assemble output table
    out = pd.concat([df[id_cols], y_star_pred], axis=1)
    if low_grade_prob is not None:
        out["p_low_grade"] = low_grade_prob
    out["retrofit_priority"] = priority

    # Sort descending by priority (higher = more urgent)
    out = out.sort_values("retrofit_priority", ascending=False)
    out.to_csv(outdir / "priority_ranked.csv", index=False)

    # Save top-k slice
    topk = out.head(args.topk).copy()
    topk.to_csv(outdir / "priority_topk.csv", index=False)

    # ------------------------------ SHAP explanations -------------------------
    if not SHAP_AVAILABLE:
        print("[warn] SHAP not installed; skipping SHAP plots. pip install shap")
        return

    # Build an input frame for SHAP on top-k
    # Use the *raw* columns that the preprocessor expects
    topk_idx = topk.index
    X_topk_raw = df.loc[topk_idx, feature_cols].copy()

    # We need a predict function that takes raw X and runs through pre+model
    def predict_fn(raw_df: pd.DataFrame):
        return reg.predict(raw_df[feature_cols])

    # Try to use fast TreeExplainer if the final estimator is tree-based (XGB/RF/HGB)
    use_kernel = True
    est_name = reg_est.__class__.__name__.lower()
    if "xgb" in est_name or "randomforest" in est_name or "histgradientboosting" in est_name:
        try:
            explainer = shap.Explainer(reg.predict, shap.maskers.Independent(X_topk_raw, max_samples=200))
            shap_values = explainer(X_topk_raw, max_evals=2000)
            use_kernel = False
        except Exception as e:
            print(f"[warn] Tree-based SHAP failed ({e}); falling back to KernelExplainer.")

    if use_kernel:
        # Background sample = 200 random rows of the full feature matrix
        bg = df[feature_cols].sample(min(200, len(df)), random_state=42)
        explainer = shap.KernelExplainer(lambda Z: predict_fn(pd.DataFrame(Z, columns=feature_cols)), bg)
        shap_values = explainer.shap_values(X_topk_raw, nsamples=200)  # keep runtime reasonable

    # Save SHAP values for top-k
    try:
        if hasattr(shap_values, "values"):
            S = pd.DataFrame(shap_values.values, index=X_topk_raw.index, columns=feature_cols)
        else:
            # KernelExplainer returns ndarray
            S = pd.DataFrame(shap_values, index=X_topk_raw.index, columns=feature_cols)
        S.to_csv(outdir / "shap_values_topk.csv", index=True)
    except Exception as e:
        print(f"[warn] Could not save SHAP values CSV: {e}")

    # Summary plot
    try:
        shap.summary_plot(shap_values, X_topk_raw, show=False, max_display=20)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary_topk.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[warn] Could not save SHAP summary plot: {e}")

    # Small manifest
    manifest = {
        "artifacts": str(pathlib.Path(args.artifacts).resolve()),
        "family": family,
        "alpha": args.alpha,
        "topk": args.topk,
        "n_rows": int(len(df)),
        "feature_cols_used": feature_cols,
        "outputs": {
            "ranked": str((outdir / "priority_ranked.csv").resolve()),
            "topk": str((outdir / "priority_topk.csv").resolve()),
            "shap_values_topk": str((outdir / "shap_values_topk.csv").resolve()),
            "shap_summary_topk": str((outdir / "shap_summary_topk.png").resolve()),
        }
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    # quiet some noisy warnings for a cleaner run
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
