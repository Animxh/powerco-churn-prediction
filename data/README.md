# Data Directory

Raw and processed data files are not included in this repository due to file size. This document describes what each file contains and where it fits in the pipeline.

---

## Files Required

### `client_data.csv`
Used in: `01_eda.ipynb`

SME customer records from PowerCo. One row per customer. Covers contract dates, consumption metrics, financial margin data, and the binary target variable `churn` (1 = churned, 0 = retained).

Rows: 14,606 | Columns: 26 | Missing values: none

---

### `price_data.csv`
Used in: `01_eda.ipynb`, `02_feature_engineering.ipynb`

Monthly energy pricing history for each customer across January–December 2015. One row per customer per month. Contains off-peak, peak, and mid-peak prices split into variable and fixed components.

Rows: 193,002 | Columns: 8

---

### `clean_data_after_eda.csv`
Used in: `02_feature_engineering.ipynb`

Output of `01_eda.ipynb`. Client data after EDA cleaning — date columns parsed, ready for feature engineering.

---

### `data_for_predictions.csv`
Used in: `03_modeling.ipynb`

Final feature-engineered dataset ready for model training. Contains the original client features plus all 29 engineered variables. Fully numeric, no nulls.

Rows: 14,606 | Columns: 63

---

## Pipeline Overview

```
client_data.csv ──┐
                  ├──► 01_eda.ipynb ──► clean_data_after_eda.csv ──► 02_feature_engineering.ipynb ──► data_for_predictions.csv ──► 03_modeling.ipynb
price_data.csv  ──┘
```

---

*All customer IDs are anonymised hashes. Data sourced from the BCG X Forage simulation.*
