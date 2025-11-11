# Carbon Cruncher – NYC Building Retrofit Prioritization

**Goal:** Triage which buildings most need retrofit investment and *why*,
using LL33 grades, LL84 benchmarking, PLUTO, and water data.

**Pipeline**
1. `scripts/join_nyc_datasets.py` – Build a clean training table.
2. `scripts/train_nyc_models.py` – Train models (RF or XGBoost).
3. `scripts/explain_priority_shap.py` – Score priority & generate SHAP explanations.

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -r requirements.txt

# Build the dataset
python scripts/join_nyc_datasets.py   --ll33 data/raw/ll33_2022_from_pdf.csv   --pluto data/raw/nyc_pluto_25v3_csv.zip   --water data/raw/Water_Consumption_in_the_City_of_New_York.csv   --ll84 data/raw/ll84_benchmarking.csv   --ll33-year 2022   --out data/processed/nyc_energy_joined.csv

# Train (choose rf or xgb)
python scripts/train_nyc_models.py   --csv data/processed/nyc_energy_joined.csv   --outdir artifacts   --model xgb

# Explain & rank
python scripts/explain_priority_shap.py   --csv data/processed/nyc_energy_joined.csv   --artifacts artifacts   --outdir artifacts/shap   --prefer full   --topk 2000
```

Artifacts:
- `artifacts/energy_star_regressor_full_<family>.joblib`
- `artifacts/metadata.json`
- `artifacts/shap/ranking_full.csv`
- `artifacts/shap/shap_summary_full.png`
- `artifacts/shap/topk_shap_long_full.csv`

## Azure Integration (high-level)
- **Azure ML**: submit jobs for join → train → explain; register model; batch score nightly.
- **Azure Web App**: package a FastAPI scoring service using saved `.joblib`.
- **Data Lake + BI**: land `ranking_full.csv` in ADLS and build Power BI dashboards.

See `docs/DATA_SOURCES.md` for column detection and joins.
