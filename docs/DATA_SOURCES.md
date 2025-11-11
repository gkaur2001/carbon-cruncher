# Data Sources

- **LL33 (Grades/Score)**: CSV (or PDF-extracted CSV). Columns commonly include
  `10_Digit_BBL`/`BBL`, `Energy_Efficiency_Grade`, `Energy_Star_1-100_Score`,
  `DOF_Gross_Square_Footage`, `Street_Number`, `Street_Name`.
- **LL84 (Benchmarking)**: CSV with
  `NYC Borough, Block and Lot (BBL)` like `1-01206-0001`, energy/EUI/fuel metrics.
  The join script converts to zero-padded 10-digit BBL.
- **PLUTO**: Zip of CSVs. Uses `bbl`, `address`, `zipcode`, `yearbuilt`,
  `numfloors`, `assesstot`, `landuse`, `latitude`, `longitude`.
- **Water**: Citywide or borough-by-year. Used for optional context features.

Joins: BBL → BIN → (address_norm, zipcode).
