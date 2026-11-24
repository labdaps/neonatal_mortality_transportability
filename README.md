# Code for "Geographic Transportability of Machine Learning Models for Neonatal Mortality Prediction Across Brazilian Municipalities"

Python code that builds the cohorts, tunes and fits the four models, and produces every table and figure of the article and its Supplementary Material.

## Contents

| Path | Content |
|---|---|
| `utils.py` | Functions shared by the notebooks (reading the raw national file, models and hyperparameter search, calibration, metrics, model comparison tests, per-municipality evaluation, SHAP, subgroups, saving). |
| `01_data.ipynb` | Cohort construction from the raw files; Table S1 (the counts behind Figure 1). |
| `02_descriptives.ipynb` | Table S2. |
| `03_modeling.ipynb` | Hyperparameter search, model fitting, calibrators and operating points; Tables S3 to S5. |
| `04_results.ipynb` | Tables 1 and 2, Figures 2 to 4; Tables S6 to S16, S19 and S20; Figures S1, S2, S4 to S6. |
| `05_sensitivity.ipynb` | Tables S17 and S18; Figure S3. |
| `raw/` | Input files (see Data). |
| `data/` | Cohorts written by `01_data.ipynb` (not distributed). |
| `models/` | Hyperparameter searches (every evaluation), fitted models, calibrators and thresholds. |
| `cache/` | Intermediate results of the heavy steps (metric batteries, per-municipality external results). |
| `results/` | Tables (`.xlsx` and `.csv`) and figures (`.pdf`, with the plotted data as `.csv`). |

## Data

The record-level files (SINASC-SIM linkages from the São Paulo Municipal Health Department, CEInfo, and from the Ministry of Health) are not distributed. `01_data.ipynb` expects them in `raw/` as `base_dn_do.csv`, `DNBRLK16.DBF` and `DNSP_DOI_2012_2020.csv`. The municipality population estimates (`raw/ibge_cnv_popsvs2024.csv`, IBGE via DATASUS) are included.

## Running

Run the notebooks in order. `03_modeling.ipynb` and `05_sensitivity.ipynb` reuse the saved hyperparameter searches and fitted models in `models/`, and `04_results.ipynb` reuses `cache/`; delete those files to recompute them (the hyperparameter search takes several hours per model). Every stochastic step uses seed 42. Figure 1 (cohort flow diagram) was drawn from the counts in Table S1.

## Environment

Python 3.12.13 with the package versions in `requirements.txt`.

## Outputs

| Item | File in `results/` |
|---|---|
| Table 1, Table 2 | `tables/Tables_1-2_main.xlsx` |
| Figure 2 | `figures/Figure_2_ROC_PR.pdf` |
| Figure 3 | `figures/Figure_3_calibration_decision.pdf` |
| Figure 4 | `figures/Figure_4_SHAP.pdf` |
| Tables S1 to S20 | `supplement_tables/Table_S<n>.xlsx` |
| Figures S1 to S6 | `supplement_figures/Figure_S<n>_<name>.pdf` |
| Full hyperparameter search history | `supplement_tables/hyperparameter_search_history.xlsx` |
| Per-municipality results, all four models | `supplement_tables/per_municipality_auroc_all_models.xlsx` |
