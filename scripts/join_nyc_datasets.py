#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, re, json, zipfile, pathlib
import pandas as pd
from typing import Optional, Tuple, Dict, List

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_by_regex(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for p in patterns:
        for c in df.columns:
            if re.search(p, c, flags=re.I):
                return c
    return None

def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_ll84_bbl(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = []
    for v in s:
        m = re.match(r"(\d+)[- ]?(\d+)[- ]?(\d+)", v)
        if m:
            boro, blk, lot = m.group(1), m.group(2), m.group(3)
            try: boro = str(int(boro))
            except Exception: boro = re.sub(r"\D", "", boro) or "0"
            blk = re.sub(r"\D", "", blk).zfill(5)
            lot = re.sub(r"\D", "", lot).zfill(4)
            out.append(f"{boro}{blk}{lot}")
        else:
            out.append(re.sub(r"\D","", v).zfill(10))
    return pd.Series(out, index=series.index)

def load_ll33(path: str, forced_year: int | None = None) -> tuple[pd.DataFrame, Dict[str,str]]:
    p = pathlib.Path(path)
    detect = {}
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            main_csv = None
            geo_csv = None
            for n in z.namelist():
                nl = n.lower()
                if nl.endswith(".csv") and any(k in nl for k in ["disclosure", "energy", "grade", "score"]):
                    main_csv = n
                if nl.endswith(".csv") and ("lookup" in nl) and any(k in nl for k in ["geojson", "geo"]):
                    geo_csv = n
            if main_csv is None:
                main_csv = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
            if main_csv is None:
                raise FileNotFoundError("No CSV found inside LL33 zip")
            with z.open(main_csv) as f:
                df = pd.read_csv(f, low_memory=False)
            if geo_csv:
                with z.open(geo_csv) as f:
                    gdf = pd.read_csv(f, low_memory=False)
                gdf = normalize_columns(gdf)
                bblg = first_present(gdf, ["10_digit_bbl","bbl"])
                if bblg and "latitude" in gdf.columns and "longitude" in gdf.columns:
                    gdf = gdf.rename(columns={bblg:"bbl"})
                    gdf["bbl"] = gdf["bbl"].astype(str).str.replace(r"\D","",regex=True).str.zfill(10)
                    df = normalize_columns(df)
                    if "10_digit_bbl" in df.columns and "bbl" not in df.columns:
                        df = df.rename(columns={"10_Digit_BBL":"bbl","10_digit_bbl":"bbl"})
                    df["bbl"] = df["bbl"].astype(str).str.replace(r"\D","",regex=True).str.zfill(10)
                    df = df.merge(gdf[["bbl","latitude","longitude"]], on="bbl", how="left")
                    detect["latitude"] = "latitude"; detect["longitude"] = "longitude"
    else:
        df = pd.read_csv(p, low_memory=False)
    df = normalize_columns(df)
    if "10_digit_bbl" in df.columns and "bbl" not in df.columns:
        df = df.rename(columns={"10_digit_bbl":"bbl"})
    if "bbl" in df.columns:
        df["bbl"] = df["bbl"].astype(str).str.replace(r"\D","",regex=True).str.zfill(10)
        detect["bbl"] = "bbl"
    for cand in ["energy_star_1-100_score","energy star score","energy_star_score","energy_star","score"]:
        if cand in df.columns:
            df["energy_star"] = coerce_numeric(df[cand]); detect["energy_star"] = cand; break
    for cand in ["energy_efficiency_grade","current_energy_grade","energy_grade","letter_grade","grade"]:
        if cand in df.columns:
            df["grade"] = df[cand].astype(str).str.upper().str.extract(r"([ABCD])", expand=False)
            detect["grade"] = cand; break
    for cand in ["dof_gross_square_footage","gross floor area - buildings (ft²)",
                 "gross floor area (ft²)","gross_floor_area_buildings_ft2","floor_area"]:
        if cand in df.columns:
            df["floor_area"] = coerce_numeric(df[cand]); detect["floor_area"]=cand; break
    addr_col = first_present(df, ["address","street address","property address","property name"])
    street_num_col = first_present(df, ["street_number","house_number","housenumber"])
    street_name_col = first_present(df, ["street_name","street","stname","streetname"])
    if addr_col is not None:
        df["address"] = df[addr_col].astype(str)
    elif street_num_col or street_name_col:
        s_num = df[street_num_col].astype(str) if street_num_col else ""
        s_name = df[street_name_col].astype(str) if street_name_col else ""
        df["address"] = (s_num + " " + s_name).str.strip()
    if "address" in df.columns:
        df["address_norm"] = (df["address"].astype(str).str.upper()
                              .str.replace(r"[^\w\s]"," ",regex=True)
                              .str.replace(r"\s+"," ",regex=True).str.strip())
    for cand in ["zipcode","zip code","postal code","postal_code","zip"]:
        if cand in df.columns:
            df["zipcode"] = df[cand].astype(str).str.extract(r"(\d{5})", expand=False); break
    if "reporting_year" not in df.columns and forced_year is not None:
        df["reporting_year"] = int(forced_year)
    return df, detect

def load_pluto(path: str) -> tuple[pd.DataFrame, Dict[str,str]]:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs: raise FileNotFoundError("No CSVs in PLUTO zip")
            dfs = []
            for name in csvs:
                with z.open(name) as f:
                    try: dfs.append(pd.read_csv(f, low_memory=False))
                    except Exception: pass
            df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p, low_memory=False)
    df = normalize_columns(df)
    detect = {}
    def map_to(new, candidates, regexes=None, numeric=False):
        src = first_present(df, candidates) or (find_by_regex(df, regexes) if regexes else None)
        if not src: return
        val = df[src]
        if new in ("bbl","bin","zipcode"):
            if new=="zipcode":
                df[new] = val.astype(str).str.extract(r"(\d{5})", expand=False)
            else:
                df[new] = val.astype(str).str.replace(r"\D","",regex=True)
        elif new=="address_norm":
            df["address"] = val.astype(str)
            df["address_norm"] = df["address"].str.upper().str.replace(r"\s+"," ",regex=True)
        elif numeric:
            df[new] = coerce_numeric(val)
        else:
            df[new] = val
        detect[new] = src
    map_to("bbl", ["bbl"])
    map_to("bin", ["bin"])
    map_to("address_norm", ["address","housenum","streetname","stname"], regexes=[r"address|street"])
    map_to("zipcode", ["zipcode","zip","zip code"])
    map_to("year_built", ["yearbuilt","year_built"] , numeric=True)
    map_to("num_floors", ["numfloors","num_floors"], numeric=True)
    map_to("assessed_value", ["assesstot","assessedtot","assessland","assesstot"] , numeric=True)
    map_to("land_use", ["landuse","land_use"])
    map_to("latitude", ["latitude"])
    map_to("longitude", ["longitude"])
    if "bbl" in df.columns:
        df["bbl"] = df["bbl"].astype(str).str.replace(r"\D","", regex=True).str.zfill(10)
    return df, detect

def load_water(path: str) -> tuple[pd.DataFrame, Dict[str,str]]:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    detect = {}
    rename_map = {
        "fiscal year":"year","fiscal_year":"year",
        "total consumption (mg)":"total_mg","total_consumption_mg":"total_mg",
        "total consumption (mgd)":"total_mgd","total_consumption_mgd":"total_mgd",
        "per capita (gallons per person per day)":"per_capita_gpd",
    }
    for old,new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old:new})
    has_borough = "borough" in df.columns
    if "year" in df.columns: df["year"] = coerce_numeric(df["year"])
    if "total_mg" not in df.columns and "total_mgd" in df.columns:
        df["total_mg"] = coerce_numeric(df["total_mgd"]) * 365.0
    keep = ["year","population","per_capita_gpd","total_mg"]
    cols = [c for c in keep if c in df.columns]
    if has_borough:
        cols = ["borough"] + cols
        bmap = {"MANHATTAN":"MN","MN":"MN","NEW YORK":"MN","BROOKLYN":"BK","BK":"BK","KINGS":"BK",
                "QUEENS":"QN","QN":"QN","BRONX":"BX","BX":"BX","STATEN ISLAND":"SI","SI":"SI","RICHMOND":"SI"}
        df["borough"] = df["borough"].astype(str).str.upper().map(lambda x: bmap.get(x, x[:2]))
    df = df[cols].copy()
    grp_keys = ["borough","year"] if has_borough else ["year"]
    df = df.groupby(grp_keys, as_index=False).mean(numeric_only=True)
    df = df.add_prefix("water_").rename(columns={"water_year":"reporting_year"})
    if has_borough: df = df.rename(columns={"water_borough":"borough"})
    for k in ["borough","year","population","per_capita_gpd","total_mg","total_mgd"]:
        if k in df.columns: detect[k]=k
    return df, detect

def load_ll84(path: str) -> tuple[pd.DataFrame, Dict[str,str]]:
    df = pd.read_csv(path, low_memory=False)
    raw_cols = df.columns.tolist()
    df = normalize_columns(df)
    detect = {}
    c_bbl = first_present(df, ["nyc borough, block and lot (bbl)","bbl","nyc_bbl","nyc borough, block and lot"])
    if c_bbl is None:
        raise ValueError("LL84 file missing BBL-like fields")
    df["bbl"] = normalize_ll84_bbl(df[c_bbl]); detect["bbl"] = c_bbl
    def pick(*names):
        for n in names:
            c = first_present(df, [n.lower()])
            if c: return c
        return None
    c_es = pick("energy star score","energy_star_score")
    if c_es: df["ll84_energy_star"] = coerce_numeric(df[c_es]); detect["ll84_energy_star"]=c_es
    c_site = pick("site eui (kbtu/ftÂ²)","site eui (kbtu/ft²)","site eui (kbtu/ft2)","site eui")
    if c_site: df["ll84_site_eui"] = coerce_numeric(df[c_site]); detect["ll84_site_eui"]=c_site
    c_src = pick("source eui (kbtu/ftÂ²)","source eui (kbtu/ft²)","source eui (kbtu/ft2)","source eui")
    if c_src: df["ll84_source_eui"] = coerce_numeric(df[c_src]); detect["ll84_source_eui"]=c_src
    elec_candidates = [c for c in df.columns if "electricity" in c and "(kwh)" in c]
    gas_candidates  = [c for c in df.columns if ("natural gas" in c and "(therms)" in c) or ("gas (therms)" in c)]
    if elec_candidates:
        df["ll84_elec_kwh"] = coerce_numeric(df[elec_candidates[0]]); detect["ll84_elec_kwh"]=elec_candidates[0]
    if gas_candidates:
        df["ll84_gas_thm"] = coerce_numeric(df[gas_candidates[0]]); detect["ll84_gas_thm"]=gas_candidates[0]
    for name, out in [("postal code","zipcode"), ("latitude","latitude"), ("longitude","longitude")]:
        c = first_present(df, [name])
        if c:
            if out=="zipcode":
                df["zipcode"] = df[c].astype(str).str.extract(r"(\d{5})", expand=False)
            else:
                df[out] = coerce_numeric(df[c])
            detect[out] = c
    keep = ["bbl","ll84_energy_star","ll84_site_eui","ll84_source_eui","ll84_elec_kwh","ll84_gas_thm","zipcode","latitude","longitude"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates(subset=["bbl"]), detect

def best_effort_merge(ll33: pd.DataFrame, pluto: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    report = {"matches_bbl":0, "matches_bin":0, "matches_addr_zip":0}
    out = ll33.copy()
    if "bbl" in out.columns and "bbl" in pluto.columns:
        out = out.merge(pluto, on="bbl", how="left", suffixes=("","_pluto"))
        probe = "year_built" if "year_built" in out.columns else None
        report["matches_bbl"] = int(out[probe].notna().sum()) if probe else 0
    if "bin" in out.columns and "bin" in pluto.columns:
        mask = out.get("year_built").isna() if "year_built" in out.columns else out.index==-1
        need = out[mask]
        if not need.empty:
            merged = need.merge(pluto, on="bin", how="left", suffixes=("","_pluto"))
            out.loc[need.index, merged.columns] = merged
            report["matches_bin"] = int(merged.get("year_built").notna().sum()) if "year_built" in merged.columns else 0
    if all(k in out.columns for k in ["address_norm","zipcode"]) and all(k in pluto.columns for k in ["address_norm","zipcode"]):
        mask = out.get("year_built").isna() if "year_built" in out.columns else out.index==-1
        need = out[mask]
        if not need.empty:
            subset_cols = ["address_norm","zipcode"] + [c for c in ["year_built","num_floors","assessed_value","land_use","bbl","bin"] if c in pluto.columns]
            subset = pluto[subset_cols]
            merged = need.merge(subset, on=["address_norm","zipcode"], how="left", suffixes=("","_pluto"))
            out.loc[need.index, merged.columns] = merged
            report["matches_addr_zip"] = int(merged.get("year_built").notna().sum()) if "year_built" in merged.columns else 0
    return out, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ll33", required=True)
    ap.add_argument("--pluto", required=True)
    ap.add_argument("--water", required=True)
    ap.add_argument("--ll84", required=False, help="Optional LL84 benchmarking CSV")
    ap.add_argument("--out", default="data/processed/nyc_energy_joined.csv")
    ap.add_argument("--save-detect-json", default="artifacts/_schema_detect.json")
    ap.add_argument("--ll33-year", type=int, default=None)
    args = ap.parse_args()

    ll33, det_ll33 = load_ll33(args.ll33, forced_year=args.ll33_year)
    pluto, det_pluto = load_pluto(args.pluto)
    water, det_water = load_water(args.water)

    merged, match_report = best_effort_merge(ll33, pluto)

    det_ll84 = {}
    if args.ll84:
        try:
            ll84, det_ll84 = load_ll84(args.ll84)
            merged = merged.merge(ll84, on="bbl", how="left")
            if {"ll84_elec_kwh","floor_area"}.issubset(merged.columns):
                merged["elec_kwh_per_sf"] = merged["ll84_elec_kwh"] / merged["floor_area"].replace({0: pd.NA})
            if {"ll84_gas_thm","floor_area"}.issubset(merged.columns):
                merged["gas_thm_per_sf"]  = merged["ll84_gas_thm"] / merged["floor_area"].replace({0: pd.NA})
        except Exception as e:
            print(f"[warn] LL84 enrichment skipped: {e}")

    if "reporting_year" in merged.columns and "reporting_year" in water.columns:
        if "borough" in water.columns and "borough" in merged.columns:
            merged = merged.merge(water, on=["borough","reporting_year"], how="left")
        else:
            merged = merged.merge(water, on=["reporting_year"], how="left")

    preferred = [
        "grade","energy_star","reporting_year",
        "borough","zipcode","address","address_norm",
        "bbl","bin",
        "floor_area","num_floors","year_built","assessed_value","land_use",
        "latitude","longitude",
        "ll84_energy_star","ll84_site_eui","ll84_source_eui","ll84_elec_kwh","ll84_gas_thm",
        "elec_kwh_per_sf","gas_thm_per_sf",
        "water_total_mg","water_population","water_per_capita_gpd",
    ]
    cols = [c for c in preferred if c in merged.columns]
    final = merged[cols].copy() if cols else merged.copy()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, index=False)

    report = {
        "rows": int(len(final)),
        "cols": final.columns.tolist(),
        "match_report": match_report,
        "detected": {"ll33": det_ll33, "pluto": det_pluto, "ll84": det_ll84, "water": det_water},
        "nulls": {c:int(final[c].isna().sum()) for c in final.columns}
    }
    print(json.dumps(report, indent=2))

    try:
        pathlib.Path(args.save_detect_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_detect_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    sys.exit(main())
