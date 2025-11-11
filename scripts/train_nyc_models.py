#!/usr/bin/env python
from __future__ import annotations

import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import joblib

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

NUMERIC_CANDIDATES = [
    "floor_area", "num_floors", "year_built", "assessed_value",
    "age", "age2", "log_floor_area", "log_assessed_value", "value_per_sf",
    "zip_median_year_built", "zip_mean_floor_area",
    "elec_kwh_per_sf", "gas_thm_per_sf",
]
CATEGORICAL_CANDIDATES = [
    "borough", "land_use_bucket", "lat_bin", "lon_bin",
]

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "reporting_year" in df.columns and "year_built" in df.columns:
        df["age"] = (df["reporting_year"] - df["year_built"]).clip(lower=0)
        df["age2"] = df["age"] ** 2
    if "floor_area" in df.columns:
        df["log_floor_area"] = np.log1p(df["floor_area"].clip(lower=0))
    if "assessed_value" in df.columns:
        df["log_assessed_value"] = np.log1p(df["assessed_value"].clip(lower=0))
    if {"assessed_value", "floor_area"}.issubset(df.columns):
        denom = df["floor_area"].replace(0, np.nan)
        df["value_per_sf"] = (df["assessed_value"] / denom).replace([np.inf, -np.inf], np.nan)
    if "land_use" in df.columns:
        lu = df["land_use"].astype(str).str.strip()
        map_land = {str(i): str(i) for i in range(1, 12)}
        map_land.update({f"{i:02d}": str(i) for i in range(1, 12)})
        lu_code = lu.map(map_land).fillna(lu)
        buckets = {"1":"1_res","2":"2_multi","3":"3_mixed","4":"4_commercial","5":"5_industrial","6":"6_transport",
                   "7":"7_utility","8":"8_public","9":"9_open","10":"10_parking","11":"11_condo"}
        df["land_use_bucket"] = lu_code.map(buckets).fillna("other")
    if {"latitude", "longitude"}.issubset(df.columns):
        df["lat_bin"] = (df["latitude"] * 100).round().astype("Int64")
        df["lon_bin"] = (df["longitude"] * 100).round().astype("Int64")
    if "zipcode" in df.columns:
        z = df["zipcode"].astype(str)
        if "year_built" in df.columns:
            df["zip_median_year_built"] = z.map(df.groupby(z, dropna=False)["year_built"].median())
        if "floor_area" in df.columns:
            df["zip_mean_floor_area"] = z.map(df.groupby(z, dropna=False)["floor_area"].mean())
    for c, hi in [("floor_area", 2_000_000), ("num_floors", 80), ("assessed_value", 5e9)]:
        if c in df.columns:
            df.loc[df[c] > hi, c] = hi
    for c in ["floor_area", "num_floors", "assessed_value", "year_built", "age", "age2",
              "elec_kwh_per_sf", "gas_thm_per_sf"]:
        if c in df.columns:
            df.loc[~np.isfinite(df[c]), c] = np.nan
    return df

def to_float_ndarray(X):
    Xdf = pd.DataFrame(X)
    Xnum = Xdf.apply(pd.to_numeric, errors="coerce")
    return Xnum.to_numpy(dtype="float64", copy=False)

def to_object_with_nan(X):
    Xdf = pd.DataFrame(X).copy()
    for c in Xdf.columns:
        Xdf[c] = Xdf[c].astype("object")
    Xdf = Xdf.replace({pd.NA: np.nan})
    try:
        from pandas import NaT
        Xdf = Xdf.replace({NaT: np.nan})
    except Exception:
        pass
    return Xdf.astype(object).to_numpy(dtype=object, copy=False)

def build_preprocessor(df: pd.DataFrame, model_type: str):
    num_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    if model_type == "xgb":
        numeric = Pipeline([("to_float", FunctionTransformer(to_float_ndarray, feature_names_out="one-to-one"))])
    else:
        numeric = Pipeline([
            ("to_float", FunctionTransformer(to_float_ndarray, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
        ])
    categorical = Pipeline([
        ("to_obj", FunctionTransformer(to_object_with_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])
    pre = ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop",
    )
    return pre, num_cols, cat_cols

def train_regressor(df: pd.DataFrame, pre, num_cols, cat_cols, model_type="rf", random_state=42):
    target = "energy_star"
    data = df.dropna(subset=[target]).copy()
    if data.empty: return None, {}
    X = data[num_cols + cat_cols]; y = data[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    if model_type == "xgb":
        if not XGB_AVAILABLE: raise RuntimeError("xgboost not installed")
        model = XGBRegressor(n_estimators=800, learning_rate=0.07, max_depth=8,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return pipe, {"rmse": rmse, "r2": r2, "n_test": int(len(y_test))}

def train_classifier(df: pd.DataFrame, pre, num_cols, cat_cols, model_type="rf", random_state=42):
    target = "grade"
    data = df.dropna(subset=[target]).copy()
    if data.empty: return None, {}
    y_letter = data[target].astype(str).str.upper().str.extract(r"([ABCD])", expand=False)
    mask = y_letter.notna()
    data = data.loc[mask].copy()
    y_letter = y_letter.loc[mask].astype(str)
    X = data[num_cols + cat_cols]
    label_to_int = {"A":0,"B":1,"C":2,"D":3}
    int_to_label = {v:k for k,v in label_to_int.items()}
    y = y_letter.map(label_to_int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=random_state)
    if model_type == "xgb":
        if not XGB_AVAILABLE: raise RuntimeError("xgboost not installed")
        model = XGBClassifier(n_estimators=800, learning_rate=0.07, max_depth=8,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              objective="multi:softprob", num_class=4, tree_method="hist", n_jobs=-1, random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=random_state)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_test_letters = y_test.map(int_to_label)
    y_pred_letters = pd.Series(y_pred).map(int_to_label)
    acc = float(accuracy_score(y_test_letters, y_pred_letters))
    f1m = float(f1_score(y_test_letters, y_pred_letters, average="macro"))
    try:
        report = classification_report(y_test_letters, y_pred_letters, output_dict=True, zero_division=0,
                                       labels=["A","B","C","D"], target_names=["A","B","C","D"])
    except TypeError:
        report = classification_report(y_test_letters, y_pred_letters, output_dict=True,
                                       labels=["A","B","C","D"], target_names=["A","B","C","D"])
    return pipe, {"accuracy": acc, "f1_macro": f1m, "n_test": int(len(y_test)), "per_class": report}

def cv_scores_regression(df, pre, num_cols, cat_cols, model_type="rf"):
    target = "energy_star"
    data = df.dropna(subset=[target]).copy()
    if data.empty: return {}
    X = data[num_cols + cat_cols]; y = data[target].astype(float)
    if model_type == "xgb" and XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=600, learning_rate=0.08, max_depth=8,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=42)
        pipe = Pipeline([("pre", pre), ("model", model)])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
        return {"xgb_rmse_cv_mean": float(np.sqrt(mse.mean()))}
    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42))])
    lin = Pipeline([("pre", pre), ("sc", StandardScaler(with_mean=False)), ("model", Ridge(alpha=1.0))])
    gb  = Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(random_state=42))])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_mse = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_squared_error")
    lin_mse = -cross_val_score(lin, X, y, cv=kf, scoring="neg_mean_squared_error")
    gb_mse  = -cross_val_score(gb, X, y, cv=kf, scoring="neg_mean_squared_error")
    return {"rf_rmse_cv_mean": float(np.sqrt(rf_mse.mean())),
            "lin_rmse_cv_mean": float(np.sqrt(lin_mse.mean())),
            "gb_rmse_cv_mean": float(np.sqrt(gb_mse.mean()))}

def cv_scores_classification(df, pre, num_cols, cat_cols, model_type="rf"):
    target = "grade"
    data = df.dropna(subset=[target]).copy()
    if data.empty: return {}
    y = data[target].astype(str).str.upper().str.extract(r"([ABCD])", expand=False)
    data = data[~y.isna()].copy()
    y = y.dropna().astype(str)
    X = data[num_cols + cat_cols].loc[y.index]
    if model_type == "xgb" and XGB_AVAILABLE:
        y_int = y.map({"A":0,"B":1,"C":2,"D":3})
        model = XGBClassifier(n_estimators=600, learning_rate=0.08, max_depth=8,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              objective="multi:softprob", num_class=4, tree_method="hist",
                              n_jobs=-1, random_state=42)
        pipe = Pipeline([("pre", pre), ("model", model)])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1 = cross_val_score(pipe, X, y_int, cv=skf, scoring="f1_macro")
        return {"xgb_f1_macro_cv_mean": float(f1.mean())}
    rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42))])
    try:
        lr_model = LogisticRegression(solver="saga", penalty="l2", class_weight="balanced", max_iter=2000)
    except Exception:
        lr_model = LogisticRegression(class_weight="balanced", max_iter=2000)
    lr = Pipeline([("pre", pre), ("sc", StandardScaler(with_mean=False)), ("model", lr_model)])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_f1 = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
        lr_f1 = cross_val_score(lr, X, y, cv=skf, scoring="f1_macro")
    return {"rf_f1_macro_cv_mean": float(rf_f1.mean()), "lr_f1_macro_cv_mean": float(lr_f1.mean())}

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = ap.parse_args()
    if args.model == "xgb" and not XGB_AVAILABLE:
        raise SystemExit("xgboost not installed. Run: pip install xgboost")
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.csv)
    if "bbl" in df.columns:
        df = df[df["bbl"].astype(str).str.fullmatch(r"\d{10}") == True]
    if "address" in df.columns:
        df = df[df["address"].notna()]
    df = engineer_features(df)
    pre, num_cols, cat_cols = build_preprocessor(df, args.model)
    reg_pipe, reg_metrics = train_regressor(df, pre, num_cols, cat_cols, model_type=args.model)
    clf_pipe, clf_metrics = train_classifier(df, pre, num_cols, cat_cols, model_type=args.model)
    cv_reg = cv_scores_regression(df, pre, num_cols, cat_cols, model_type=args.model)
    cv_clf = cv_scores_classification(df, pre, num_cols, cat_cols, model_type=args.model)
    meta = {
        "model_family": args.model,
        "features_numeric": num_cols,
        "features_categorical": cat_cols,
        "targets": {"regression": "energy_star", "classification": "grade"},
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
        "cv_regression": cv_reg,
        "cv_classification": cv_clf,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    if reg_pipe is not None:
        joblib.dump(reg_pipe, outdir / f"energy_star_regressor_full_{args.model}.joblib")
    if clf_pipe is not None:
        joblib.dump(clf_pipe, outdir / f"grade_classifier_full_{args.model}.joblib")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
