# PowerCo SME Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end churn prediction for an energy utility's SME segment — covering EDA, feature engineering, Random Forest modelling, and a stakeholder-ready executive summary. Built as part of the BCG X Data Science Job Simulation on Forage.

---

## Project Summary

PowerCo was losing SME customers and wanted to know why — and more importantly, who was next. This project works through a BCG consulting simulation: raw client and pricing data goes in, a predictive churn model and an executive summary come out.

Four Jupyter notebooks cover the full pipeline. Each one builds directly on the last.

---

## Key Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.682** |
| Recall | **52.9%** |
| Precision | 16.6% |
| F1 Score | 0.253 |
| Churn Rate (dataset) | 9.7% |
| Customers Analysed | 14,606 |
| Features Engineered | 29 new features |

In plain terms: the model catches more than half of all churners before they leave. That gives the retention team a ranked list to work from instead of guessing.

---

## Repository Structure

```
powerco-churn-prediction/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation & validation
│   ├── 03_modeling.ipynb             # Random Forest training & evaluation
│   └── 04_executive_summary.ipynb    # Summary for stakeholders
│
├── data/
│   └── README.md                     # Data source descriptions
│
├── reports/
│   └── BCG_PowerCo_Executive_Summary.pdf  # Steering committee slide
│
├── src/
│   ├── features.py                   # Feature engineering functions
│   └── evaluate.py                   # Model evaluation utilities
│
├── assets/
│   └── pipeline_overview.png         # Visual pipeline diagram
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to exclude from git
├── LICENSE                           # MIT License
└── README.md                         # You are here
```

---

## Project Walkthrough

### Phase 1 — Exploratory Data Analysis

14,606 SME customer records and 193,002 monthly price observations (Jan–Dec 2015). A few things came out of it that changed how the rest of the project was approached.

The most important: only 9.7% of customers actually churned. That sounds fine until you realise a model that predicts "no churn" for every single customer scores 90.3% accuracy — and catches nobody. Every modelling decision after this point was made with that number in mind.

The second finding was less expected: raw price variables barely correlated with churn. Price *levels* are not really the issue. The *change* in price over time, and how much margin the customer generates, are what actually predict who leaves. The third finding was more textbook — consumption is heavily right-skewed, a few industrial customers dominate the distribution, and log transformation is needed before the model can use it cleanly. Sales channel also turned out to matter: some acquisition channels produce customers who churn at noticeably higher rates than others.

### Phase 2 — Feature Engineering

29 new features across five categories. Here is the reasoning behind each group.

**Date-based features** extract information locked inside contract timestamps. How long a customer has been active, whether their product was recently modified, and how far in advance they renewed their contract all turn out to be meaningful signals. A customer in the last six months of a contract who just had their product tweaked is a different risk profile from a seven-year customer who renewed early.

**Consumption features** capture behavioural changes. `cons_ratio_last_to_avg` — last month's usage divided by the 12-month average — is particularly useful. A sharp drop suggests the customer may have already moved some of their load elsewhere.

**Price variation composites** compress 18 raw price-change columns into a handful of usable signals: total price volatility over the year, whether that volatility accelerated recently (`price_acceleration_ratio`), and the December-vs-January off-peak swing that was the original hypothesis going into the project.

**Financial and margin features** are where the real action turned out to be. `net_margin_per_kwh` (importance 0.118) and `margin_gross_pow_ele` (0.106) became the top two model features. Low-margin customers are also the ones who leave. That finding reshaped the business recommendations.

**Categorical encoding** handled `channel_sales`, `origin_up`, and `has_gas` using `pd.factorize()` — no sklearn dependency, and the mapping is fully transparent.

### Phase 3 — Modelling & Evaluation

A Random Forest classifier trained with 1,000 trees, max depth capped at 5, and `class_weight='balanced'`. Each of those choices has a reason.

Capping depth at 5 keeps individual trees shallow and weak on purpose. A single deep tree would memorise the training data. A thousand shallow ones that each learn a fragment of the pattern combine into something far more reliable. `class_weight='balanced'` is the most consequential setting: without it, the model learns to predict "no churn" constantly, scores 90% accuracy, and is completely useless. The balanced weighting penalises missed churners more heavily, which is exactly what the business problem requires. `stratify=y` in the train/test split ensures both halves of the data contain the same 9.7% churn proportion — a small detail that matters a lot for evaluation reliability.

Accuracy is not the headline metric here, because it is misleading on imbalanced data. Recall, precision, F1, and ROC-AUC each tell a different piece of the story about how the model actually performs on the minority class.

**Top 5 features by importance:**

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `margin_net_pow_ele` | 0.118 | Net margin on electricity — low-margin customers are most at risk |
| 2 | `margin_gross_pow_ele` | 0.106 | Gross margin confirms the margin-churn link |
| 3 | `origin_up` | 0.054 | Acquisition channel shapes long-term loyalty |
| 4 | `months_activ` | 0.053 | Tenure — newer customers churn more |
| 5 | `cons_12m` | 0.051 | Annual consumption — lower usage means lower switching cost |

### Phase 4 — Executive Summary

The final output is a single-slide PDF built for a steering committee audience. No model jargon, no confusion matrices. Just the headline churn rate, what the model found in business terms, and four next steps the commercial team can act on immediately.

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Animxh/powerco-churn-prediction.git
cd powerco-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the data files**

Place the following CSV files in the `data/` directory (not included due to file size):
- `client_data.csv` — SME customer records
- `price_data.csv` — Monthly pricing history
- `clean_data_after_eda.csv` — EDA-cleaned client data
- `data_for_predictions.csv` — Final feature-engineered dataset

**4. Run the notebooks in order**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Methodology Notes

**Why Random Forest and not XGBoost?** The simulation specified Random Forest. Gradient boosted models (XGBoost, LightGBM) generally outperform it on structured tabular data, and running a comparison in Phase 2 would be a reasonable next step.

**Why not SMOTE?** `class_weight='balanced'` does the same job more cleanly. SMOTE applied before splitting can leak synthetic samples from training into the test set — a subtle but real problem. If SMOTE is used in a future iteration, it needs to live inside a proper sklearn Pipeline so it only sees training data.

**On the precision-recall trade-off:** 16.6% precision means five out of six flagged customers are false positives. Whether that is acceptable depends on the outreach cost-to-contract-value ratio. For multi-year SME energy contracts, the numbers usually favour catching more churners even at the cost of some wasted calls — but that calculation should be validated against real contract values before deploying at scale.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | ML model & evaluation |
| Matplotlib | Visualisation |
| Seaborn | Statistical plots |
| ReportLab | PDF generation |
| Jupyter | Interactive notebooks |

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Certificate

Completed as part of the **BCG X Data Science Job Simulation** on [Forage](https://www.theforage.com/).

| | |
|---|---|
| **Issued** | March 15, 2026 |
| **Completed by** | Animesh Sen |
| **Issuer** | BCG X / Forage |
| **Enrolment Code** | `gpfQ7dtg3RLvtny6q` |
| **User Code** | `aD7kKW4f8ZPWygnJy` |

Tasks completed: Business Framing, EDA & Data Cleaning, Feature Engineering, Modelling & Evaluation, Insights & Recommendations.

Certificate PDF: [`reports/BCG_Forage_Certificate.pdf`](reports/BCG_Forage_Certificate.pdf)  
Verifiable at [theforage.com](https://www.theforage.com/) using the codes above.

---

## Author

**Animesh Sen**  
Process Associate → Aspiring Data Analyst  
[LinkedIn](https://linkedin.com/in/animeshsen03) · [GitHub](https://github.com/Animxh)

---

*All data used in this project is anonymised and sourced from the BCG X Forage simulation.*
