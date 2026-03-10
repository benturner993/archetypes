# Bupa Dental — Practice Archetypes Analysis

A two-stage analytical pipeline that:
1. Runs exploratory data analysis across five operational dimensions
2. Classifies every practice into a **4 × 4 Size × Model archetype matrix**

---

## Project files

| File | Purpose |
|------|---------|
| `master.csv` | Input data — one row per practice (replace with real data) |
| `eda_practices.ipynb` | Exploratory analysis across five operational dimensions |
| `01_rules_based.ipynb` | Approach 1: deterministic threshold classification |
| `02_clustering.ipynb` | Approach 2: unsupervised K-Means classification |
| `03_modelling.ipynb` | Approach 3: XGBoost income modelling + outlier detection |
| `archetype_analysis.py` | Standalone script that runs all three approaches in sequence |
| `archetypes_rules.csv` | Output from `01_rules_based.ipynb` |
| `archetypes_clustering.csv` | Output from `02_clustering.ipynb` |
| `archetypes_modelling.csv` | Output from `03_modelling.ipynb` |
| `master_archetypes.csv` | Combined output from `archetype_analysis.py` |
| `create_data.py` | Synthetic data generator (development/testing only) |
| `create_metadata.py` | Schema reference (development/testing only) |

---

## Quickstart with real data

### 1 · Replace the input file

Drop your real export in place of the synthetic `master.csv`. The file must be a CSV with a header row matching the schema below — column names are case-sensitive.

```
practicekey, practicecode, practicename, brandname, practiceid,
region, campus, postcode, numberofsurgeries, numberofchairs,
status, acquiredon, nps, countof_dentistid, practise_code,
practise_name, practice_code_and_name, unique_staff_ids, clinical_roles,
position_dentist, position_dental_nurse, position_receptionist,
position_hygienist, position_practice_manager,
contractualhours_dentist, contractualhours_dental_nurse,
contractualhours_receptionist, contractualhours_hygienist,
contractualhours_practice_manager, monthkey_x, active, nov,
period, latest_month, nooftreatmentitems,
nooftreatmentitems_private_standard, nooftreatmentitems_private_referral,
nooftreatmentitems_nhs_standard, nooftreatmentitems_nhs_referral,
noofpatients_private_standard, noofpatients_private_referral,
noofpatients_nhs_standard, noofpatients_nhs_referral,
chargeprice_private_standard, chargeprice_private_referral,
chargeprice_nhs_standard, chargeprice_nhs_referral,
monthkey_y, privateincome, uda, latest_month_y, countof_snareid
```

Any extra columns in your export are ignored. Missing columns will default to zero — check the **Data notes** section below before running.

### 2 · Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Tested with: `pandas 2.0.3` · `numpy 1.21.2` · `matplotlib 3.7.5` · `seaborn 0.11.1` · `scikit-learn 1.3.2` · `xgboost 2.1.4`

### 3 · Choose how to run the analysis

**Option A — Individual approach notebooks (recommended for exploration)**

Run each notebook independently in Jupyter. They are self-contained and can be run in any order, though `02_clustering.ipynb` and `03_modelling.ipynb` will optionally load `archetypes_rules.csv` produced by `01_rules_based.ipynb` for comparison views.

```bash
jupyter notebook 01_rules_based.ipynb
jupyter notebook 02_clustering.ipynb
jupyter notebook 03_modelling.ipynb
```

Each notebook saves its own output CSV:

| Notebook | Output file | Key columns |
|----------|-------------|-------------|
| `01_rules_based.ipynb` | `archetypes_rules.csv` | `archetype_size`, `archetype_model`, `archetype_rules` |
| `02_clustering.ipynb` | `archetypes_clustering.csv` | `cluster_size_id`, `cluster_model_id`, `archetype_size`, `archetype_model` |
| `03_modelling.ipynb` | `archetypes_modelling.csv` | `predicted_income`, `income_residual`, `residual_z`, `predicted_performance_tier` |

**Option B — Run all three at once via the script**

```bash
python archetype_analysis.py
```

Writes `master_archetypes.csv` with all columns from all three approaches combined.

### 4 · Open the EDA notebook

```bash
jupyter notebook eda_practices.ipynb
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**). This notebook covers the five exploratory dimensions only and does not depend on any of the archetype approach notebooks.

---

## Archetype framework

### Size bands

| Band | Primary signal | Secondary override |
|------|---------------|--------------------|
| Small / Foundation | ≤ 3 surgeries | — |
| Medium / Core | 4–5 surgeries | — |
| Large / Advanced | 6–7 surgeries | — |
| Flagship | ≥ 8 surgeries | ≥ 6 surgeries AND ≥ 15 staff |

### Model bands

NHS share of income is calculated as `(uda × £28) / (uda × £28 + privateincome)`.  
Thresholds are percentile-anchored to the data distribution so segments remain balanced when the NHS/private split shifts with real data.

| Band | Signal |
|------|--------|
| NHS Led | NHS share ≥ P75 of portfolio |
| Balanced Mixed | NHS share P50–P75 |
| Private Led Mixed | NHS share P25–P50 |
| Specialist / Referral Hub | NHS share < P25 AND hygienist present AND high private income per chair; OR NHS share < P10 |

> **N/A zones**: NHS Led cannot be Large/Advanced or Flagship. Practices that fall into those cells are automatically reclassified to Balanced Mixed.

### What improves with real data

The synthetic `master.csv` has several columns set to zero that will unlock richer classification once real data is loaded:

| Column | Effect when populated |
|--------|-----------------------|
| `nooftreatmentitems_nhs_standard` / `_private_standard` | Direct NHS vs private treatment split replaces UDA proxy |
| `nooftreatmentitems_private_referral` / `_nhs_referral` | Real referral rate drives Specialist/Referral Hub flag |
| `countof_snareid` | Specialist network activity confirms referral hub classification |
| `chargeprice_private_referral` | Revenue per referral item feeds model clustering |
| `nps` | Constant in synthetic data; with real variance it becomes a modelling target |
| `clinical_roles` | Enables clinical-role mix as a direct workforce signal |

To activate these signals, update the `_rules_model()` function thresholds in `archetype_analysis.py` once you have verified the distributions in your real data using the EDA notebook.

---

## Tuning the classification

### Rules-based thresholds

Open `archetype_analysis.py` and edit the threshold functions directly:

```python
# Size — adjust surgery cut-offs to match your portfolio distribution
def _rules_size(df):
    ...
    size[s <= 3]          = "Small/Foundation"   # change 3 to your P25 surgery count
    size[(s >= 4) & (s <= 5)] = "Medium/Core"
    size[(s >= 6) & (s <= 7)] = "Large/Advanced"
    size[s >= 8]           = "Flagship"

# Model — adjust NHS share percentile bands
def _rules_model(df):
    q10 = df["nhs_share"].quantile(0.10)   # change percentile anchors if needed
    q25 = df["nhs_share"].quantile(0.25)
    ...
```

### NHS value per UDA

The `£28 per UDA` rate used to estimate NHS income is set at the top of the script:

```python
NHS_VALUE_PER_UDA = 28.0   # update to match current contracted rate
```

### Clustering

The number of clusters (default 4 for both size and model) is controlled by the `n_clusters` argument in `apply_clustering()`. Increase to 5 if you need finer granularity, or reduce to 3 for a simpler view.

---

## EDA notebook sections (`eda_practices.ipynb`)

| Section | What it covers |
|---------|---------------|
| 1 · Practice Size, Growth & Demand | Surgery/chair/staff distributions, acquisition growth, regional demand |
| 2 · NHS vs Private Mix & Revenue | Income share distributions, regional stacked revenue, practice segments |
| 3 · Standard vs Specialist Referral | Referral rate distribution, NHS vs private split, regional comparison |
| 4 · Workforce Mix & Role Design | Headcount by role, nurse:dentist ratio, hygienist penetration |
| 5 · Capacity & Productivity | Treatment items/income per surgery, UDAs per dentist, whitespace sizing |
| Summary Dashboard | Portfolio-level KPI table |

## Approach notebook summaries

| Notebook | Approach | When to use |
|----------|----------|-------------|
| `01_rules_based.ipynb` | Hard-coded thresholds on surgeries and NHS share | Default classification; easy to explain to stakeholders; stable across data refreshes |
| `02_clustering.ipynb` | K-Means on standardised features (k=4) | Validate rules; discover natural groupings; explore if real data reveals different structure |
| `03_modelling.ipynb` | XGBoost income model + z-score outlier flagging | Identify over- and under-performers; understand which features drive income; prioritise interventions |
