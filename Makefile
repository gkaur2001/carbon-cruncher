PY=python

DATA_PROCESSED=data/processed/nyc_energy_joined.csv
ART=artifacts
SHAP_OUT=artifacts/shap

data:
	@$(PY) scripts/join_nyc_datasets.py \	  --ll33 data/raw/ll33_2022_from_pdf.csv \	  --pluto data/raw/nyc_pluto_25v3_csv.zip \	  --water data/raw/Water_Consumption_in_the_City_of_New_York.csv \	  --ll84 data/raw/ll84_benchmarking.csv \	  --ll33-year 2022 \	  --out $(DATA_PROCESSED)

train:
	@$(PY) scripts/train_nyc_models.py \	  --csv $(DATA_PROCESSED) \	  --outdir $(ART) \	  --model xgb

explain:
	@$(PY) scripts/explain_priority_shap.py \	  --csv $(DATA_PROCESSED) \	  --artifacts $(ART) \	  --outdir $(SHAP_OUT) \	  --prefer full \	  --topk 2000
