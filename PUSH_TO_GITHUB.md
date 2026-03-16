# How to Push This to GitHub

Everything is ready. Follow these steps and the repo will be live in under five minutes.

---

## Step 1 — Create the repo on GitHub

Go to github.com/new and log in. Fill in the following:

- **Repository name:** powerco-churn-prediction
- **Description:** End-to-end SME churn prediction — EDA, feature engineering & Random Forest modelling (BCG X simulation)
- **Visibility:** Public
- Leave all the checkboxes unticked — the zip already has a README, .gitignore, and LICENSE

Click **Create repository** and leave the page open.

---

## Step 2 — Unzip the folder

Unzip `powerco-churn-prediction-v3.zip` somewhere easy to find, like your Desktop.

---

## Step 3 — Copy your notebooks in

Put your Jupyter notebooks inside the `notebooks/` folder:

- `Task_2_-_eda_starter.ipynb` → rename to `01_eda.ipynb`
- `feature_engineering_bcg.ipynb` → rename to `02_feature_engineering.ipynb`
- `modeling_bcg.ipynb` → rename to `03_modeling.ipynb`

---

## Step 4 — Open a terminal and run these commands

```bash
cd Desktop/powerco-churn-prediction

git init
git add .
git commit -m "Initial commit — BCG PowerCo churn prediction project"
git branch -M main
git remote add origin https://github.com/Animxh/powerco-churn-prediction.git
git push -u origin main
```

When Git asks for your username, enter `Animxh`.

When it asks for your password, do not use your GitHub account password — GitHub stopped accepting those. You need a Personal Access Token instead. To get one: go to GitHub → profile photo → Settings → Developer Settings → Personal access tokens → Tokens (classic) → Generate new token. Tick the `repo` scope, generate, copy the token, and paste it as your password. You only need to do this once.

---

## Step 5 — Verify and pin

Refresh the GitHub page from Step 1. You should see the full folder structure and the formatted README with badges and results tables.

Then go to your GitHub profile, click **Customize your pins**, and add this repo so it shows up first.

---

Your repo will be live at:
https://github.com/Animxh/powerco-churn-prediction
