"""
archetype_analysis.py
─────────────────────
Assigns every practice in master.csv to a position in the Archetypes Progress
framework: a 4×4 matrix of Size × Model.

Size bands   : Small/Foundation | Medium/Core | Large/Advanced | Flagship
Model bands  : NHS Led | Balanced Mixed | Private Led Mixed | Specialist/Referral Hub

Three labelling strategies are implemented:
  1. apply_rules()      – deterministic heuristics derived from the framework
  2. apply_clustering() – unsupervised K-Means (size and model independently)
  3. apply_modeling()   – regression-based performance tier + outlier flagging

N/A zone enforcement (per framework slide):
  NHS Led practices cannot be Large/Advanced or Flagship — they are
  reclassified to Balanced Mixed if the size rule fires at those levels.

Data notes (synthetic master.csv):
  • nooftreatmentitems_nhs_standard / _referral columns are all zero.
    NHS activity is proxied via UDA counts × £28 (standard UDA rate).
  • countof_snareid and chargeprice_private_referral are all zero.
    Specialist/Referral Hub is proxied via hygienist presence + private
    income intensity.
  • nps is constant (47.7) across all practices and is excluded from
    modelling targets; total estimated income is used instead.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NHS_VALUE_PER_UDA = 28.0          # £ per UDA (standard proxy)
NA_ZONE_PAIRS = {                 # (model, size) combinations that are invalid
    ("NHS Led", "Large/Advanced"),
    ("NHS Led", "Flagship"),
}

SIZE_LABELS  = ["Small/Foundation", "Medium/Core", "Large/Advanced", "Flagship"]
MODEL_LABELS = ["NHS Led", "Balanced Mixed", "Private Led Mixed", "Specialist/Referral Hub"]


# ══════════════════════════════════════════════════════════════════════════════
# 0 · LOAD & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_engineer(path: str = "master.csv") -> pd.DataFrame:
    """Load master.csv and compute derived features used by all three approaches."""
    df = pd.read_csv(path)

    # ── Income ────────────────────────────────────────────────────────────────
    df["nhs_income_est"]   = df["uda"].fillna(0) * NHS_VALUE_PER_UDA
    df["private_income"]   = df["privateincome"].fillna(0)
    df["total_income_est"] = df["private_income"] + df["nhs_income_est"]

    df["nhs_share"] = np.where(
        df["total_income_est"] > 0,
        df["nhs_income_est"] / df["total_income_est"],
        np.nan,
    )

    # ── Activity ──────────────────────────────────────────────────────────────
    df["items_per_surgery"]  = df["nooftreatmentitems"] / df["numberofsurgeries"].replace(0, np.nan)
    df["income_per_surgery"] = df["total_income_est"]   / df["numberofsurgeries"].replace(0, np.nan)
    df["uda_per_dentist"]    = df["uda"]                / df["position_dentist"].replace(0, np.nan)

    # ── Specialist proxy (hygienist presence + private intensity) ─────────────
    df["has_hygienist"]            = df["position_hygienist"] > 0
    df["private_income_per_chair"] = df["private_income"] / df["numberofchairs"].replace(0, np.nan)

    # High private intensity = top quartile of private income per chair
    hi_private_thresh = df["private_income_per_chair"].quantile(0.75)
    df["high_private_intensity"] = df["private_income_per_chair"] >= hi_private_thresh

    # Specialist flag: hygienist present AND high private intensity
    df["specialist_flag"] = df["has_hygienist"] & df["high_private_intensity"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: N/A zone enforcement
# ══════════════════════════════════════════════════════════════════════════════

def enforce_na_zones(size_col: pd.Series, model_col: pd.Series) -> pd.Series:
    """
    Return a corrected model_col where invalid (model, size) pairs are
    reclassified to 'Balanced Mixed'.
    """
    model_out = model_col.copy()
    for model_val, size_val in NA_ZONE_PAIRS:
        mask = (model_col == model_val) & (size_col == size_val)
        model_out[mask] = "Balanced Mixed"
    return model_out


# ══════════════════════════════════════════════════════════════════════════════
# 1 · RULES-BASED APPROACH
# ══════════════════════════════════════════════════════════════════════════════

def _rules_size(df: pd.DataFrame) -> pd.Series:
    """
    Size classification thresholds calibrated to the data distribution:
      surgeries  : 2–9  (median 6)
      staff ids  : 7–17 (median 12)

    Small/Foundation  : ≤ 3 surgeries
    Medium/Core       : 4–5 surgeries
    Large/Advanced    : 6–7 surgeries
    Flagship          : ≥ 8 surgeries  OR  (≥ 6 surgeries AND staff ≥ 15)
    """
    s   = df["numberofsurgeries"]
    st  = df["unique_staff_ids"]
    size = pd.Series("Medium/Core", index=df.index)

    size[s <= 3]                         = "Small/Foundation"
    size[(s >= 4) & (s <= 5)]            = "Medium/Core"
    size[(s >= 6) & (s <= 7)]            = "Large/Advanced"
    size[s >= 8]                         = "Flagship"
    # Staff override: large footprint with high staffing → upgrade to Flagship
    size[(s >= 6) & (st >= 15)]          = "Flagship"

    return size


def _rules_model(df: pd.DataFrame) -> pd.Series:
    """
    Model classification using NHS share of income and specialist proxy.

    Thresholds are percentile-anchored to the synthetic data to ensure
    even segment distribution when real NHS treatment item data is absent:
      NHS Led                  : nhs_share ≥ Q75  (~top quartile)
      Balanced Mixed           : nhs_share Q50–Q75
      Private Led Mixed        : nhs_share Q25–Q50
      Specialist/Referral Hub  : nhs_share < Q25  AND specialist_flag
                                 OR  nhs_share < Q10  (extreme private skew)
    """
    q10 = df["nhs_share"].quantile(0.10)
    q25 = df["nhs_share"].quantile(0.25)
    q50 = df["nhs_share"].quantile(0.50)
    q75 = df["nhs_share"].quantile(0.75)

    model = pd.Series("Balanced Mixed", index=df.index)

    model[df["nhs_share"] >= q75]                                  = "NHS Led"
    model[(df["nhs_share"] >= q50) & (df["nhs_share"] < q75)]     = "Balanced Mixed"
    model[(df["nhs_share"] >= q25) & (df["nhs_share"] < q50)]     = "Private Led Mixed"
    # Specialist: bottom quartile NHS share + specialist signals, or extreme private skew
    specialist_mask = (
        ((df["nhs_share"] < q25) & df["specialist_flag"]) |
        (df["nhs_share"] < q10)
    )
    model[specialist_mask] = "Specialist/Referral Hub"

    return model


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      archetype_size_rules   – Size band from heuristic thresholds
      archetype_model_rules  – Model band from heuristic thresholds (NA zones enforced)
      archetype_rules        – Combined 'Size | Model' label
    """
    df = df.copy()
    size  = _rules_size(df)
    model = _rules_model(df)
    model = enforce_na_zones(size, model)

    df["archetype_size_rules"]  = size
    df["archetype_model_rules"] = model
    df["archetype_rules"]       = size + " | " + model
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2 · CLUSTERING-BASED APPROACH
# ══════════════════════════════════════════════════════════════════════════════

def _map_clusters_to_labels(
    df: pd.DataFrame,
    cluster_col: str,
    ordering_feature: str,
    labels,
    ascending: bool = True,
) -> pd.Series:
    """
    Sort cluster centroids by `ordering_feature` and map to ordered `labels`.
    ascending=True → label[0] is the cluster with the smallest centroid value.
    """
    centroid_means = (
        df.groupby(cluster_col)[ordering_feature]
        .mean()
        .sort_values(ascending=ascending)
    )
    cluster_to_label = {
        cluster_id: labels[rank]
        for rank, cluster_id in enumerate(centroid_means.index)
    }
    return df[cluster_col].map(cluster_to_label)


def apply_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      cluster_size_id      – Raw K-Means cluster id (0–3) for size
      cluster_model_id     – Raw K-Means cluster id (0–3) for model
      archetype_size_clust – Size label mapped from cluster centroids
      archetype_model_clust– Model label mapped from cluster centroids (NA zones enforced)
      archetype_clust      – Combined label
    """
    df = df.copy()
    scaler = StandardScaler()

    # ── Size clustering ───────────────────────────────────────────────────────
    size_features = [
        "numberofsurgeries",
        "unique_staff_ids",
        "contractualhours_dentist",
    ]
    X_size = scaler.fit_transform(df[size_features].fillna(0))
    km_size = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["cluster_size_id"] = km_size.fit_predict(X_size)

    size_clust = _map_clusters_to_labels(
        df, "cluster_size_id", "numberofsurgeries", SIZE_LABELS, ascending=True
    )

    # ── Model clustering ──────────────────────────────────────────────────────
    # nhs_share descending → highest NHS share = NHS Led (index 0 after ascending sort)
    model_features = [
        "nhs_share",
        "private_income",
        "nhs_income_est",
        "items_per_surgery",
        "position_hygienist",
        "private_income_per_chair",
    ]
    X_model = scaler.fit_transform(df[model_features].fillna(0))
    km_model = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    df["cluster_model_id"] = km_model.fit_predict(X_model)

    # Order by nhs_share descending: cluster with highest nhs_share → NHS Led
    model_clust = _map_clusters_to_labels(
        df, "cluster_model_id", "nhs_share", MODEL_LABELS, ascending=False
    )

    model_clust = enforce_na_zones(size_clust, model_clust)

    df["archetype_size_clust"]  = size_clust
    df["archetype_model_clust"] = model_clust
    df["archetype_clust"]       = size_clust + " | " + model_clust
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3 · MODELLING-BASED APPROACH
# ══════════════════════════════════════════════════════════════════════════════

def _encode_region(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode region; drop first to avoid dummy trap."""
    dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True, dtype=float)
    return pd.concat([df, dummies], axis=1)


def apply_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      predicted_income           – OLS-predicted total income
      income_residual            – Actual − predicted (£)
      predicted_performance_tier – 'High Outlier' / 'Expected' / 'Low Outlier'
                                   based on standardised residual (|z| > 2)

    Also prints:
      - Linear Regression: coefficients + R2 for total income
      - XGBoost: feature importances for total income
      - Outlier summary

    Note: NPS is constant in the synthetic data and cannot be modelled.
    total_income_est is used as the performance target instead.
    """
    df = df.copy()
    df = _encode_region(df)

    region_dummies = [c for c in df.columns if c.startswith("region_")]
    base_features  = [
        "numberofsurgeries",
        "unique_staff_ids",
        "uda",
        "position_hygienist",
        "contractualhours_dentist",
    ] + region_dummies

    X = df[base_features].fillna(0).values
    y = df["total_income_est"].values

    # ── Linear Regression ────────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr     = lr.predict(X)
    r2_lr         = r2_score(y, y_pred_lr)
    cv_r2_lr      = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()

    print("\n" + "-" * 60)
    print("LINEAR REGRESSION -- Total income drivers")
    print("-" * 60)
    coef_df = (
        pd.DataFrame({"feature": base_features, "coefficient": lr.coef_})
        .sort_values("coefficient", key=abs, ascending=False)
    )
    print(coef_df.to_string(index=False))
    print(f"\nIntercept : £{lr.intercept_:,.0f}")
    print(f"R²        : {r2_lr:.3f}  (5-fold CV R²: {cv_r2_lr:.3f})")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb.fit(X, y)
    cv_r2_xgb = cross_val_score(xgb, X, y, cv=5, scoring="r2").mean()

    print("\n" + "-" * 60)
    print("XGBOOST -- Feature importances for total income")
    print("-" * 60)
    imp_df = (
        pd.DataFrame({"feature": base_features, "importance": xgb.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print(imp_df.to_string(index=False))
    print(f"\n5-fold CV R2 : {cv_r2_xgb:.3f}")

    # ── Outlier detection ────────────────────────────────────────────────────
    # Use XGBoost predictions as the performance baseline
    y_pred_rf    = xgb.predict(X)
    residuals    = y - y_pred_rf
    resid_z      = (residuals - residuals.mean()) / residuals.std()

    tier = pd.Series("Expected", index=df.index)
    tier[resid_z >  2] = "High Outlier"
    tier[resid_z < -2] = "Low Outlier"

    y_pred_xgb = y_pred_rf  # alias for clarity in downstream code
    df["predicted_income"]            = np.round(y_pred_xgb, 2)
    df["income_residual"]             = np.round(residuals, 2)
    df["predicted_performance_tier"]  = tier

    outlier_summary = tier.value_counts()
    print("\n" + "-" * 60)
    print("OUTLIER SUMMARY (|z-residual| > 2)")
    print("-" * 60)
    print(outlier_summary.to_string())

    high_out = df[df["predicted_performance_tier"] == "High Outlier"][
        ["practicename", "archetype_size_rules", "archetype_model_rules",
         "total_income_est", "predicted_income", "income_residual"]
    ].head(10)
    if len(high_out):
        print("\nTop High Outliers (practices punching above their predicted income):")
        print(high_out.to_string(index=False))

    low_out = df[df["predicted_performance_tier"] == "Low Outlier"][
        ["practicename", "archetype_size_rules", "archetype_model_rules",
         "total_income_est", "predicted_income", "income_residual"]
    ].head(10)
    if len(low_out):
        print("\nTop Low Outliers (practices underperforming their predicted income):")
        print(low_out.to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4 · CROSSTAB OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def print_crosstabs(df: pd.DataFrame) -> None:
    """Print 16-cell crosstabs for all three labelling approaches."""

    size_order  = SIZE_LABELS
    model_order = MODEL_LABELS

    approaches = [
        ("Rules-Based",   "archetype_size_rules",  "archetype_model_rules"),
        ("Clustering",    "archetype_size_clust",   "archetype_model_clust"),
    ]

    for label, size_col, model_col in approaches:
        print("\n" + "-" * 70)
        print(f"PRACTICE COUNT CROSSTAB -- {label} Approach")
        print("  Rows = Size  |  Columns = Model")
        print("-" * 70)
        ct = pd.crosstab(
            df[size_col],
            df[model_col],
            margins=True,
            margins_name="TOTAL",
        ).reindex(index=size_order + ["TOTAL"], columns=model_order + ["TOTAL"], fill_value=0)
        print(ct.to_string())

    # Performance tier breakdown within rules archetypes
    print("\n" + "-" * 70)
    print("PERFORMANCE TIER BREAKDOWN WITHIN RULES ARCHETYPES")
    print("-" * 70)
    pt = pd.crosstab(
        df["archetype_size_rules"] + " | " + df["archetype_model_rules"],
        df["predicted_performance_tier"],
        margins=True,
        margins_name="TOTAL",
    )
    print(pt.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 5 · ARCHETYPE PROFILE SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def print_archetype_profiles(df: pd.DataFrame) -> None:
    """Summarise key metrics for each rules-based size × model cell."""
    print("\n" + "-" * 70)
    print("ARCHETYPE PROFILES -- Mean key metrics (Rules-Based)")
    print("-" * 70)
    profile = (
        df.groupby(["archetype_size_rules", "archetype_model_rules"], observed=True)
        .agg(
            n=("practicekey", "count"),
            avg_surgeries=("numberofsurgeries", "mean"),
            avg_staff=("unique_staff_ids", "mean"),
            avg_private_income=("private_income", "mean"),
            avg_nhs_income=("nhs_income_est", "mean"),
            avg_total_income=("total_income_est", "mean"),
            avg_nhs_share_pct=("nhs_share", lambda x: x.mean() * 100),
            avg_treatment_items=("nooftreatmentitems", "mean"),
            avg_uda=("uda", "mean"),
            pct_high_outlier=(
                "predicted_performance_tier",
                lambda x: (x == "High Outlier").mean() * 100,
            ),
        )
        .round(1)
        .sort_index()
    )
    print(profile.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(input_path: str = "master.csv", output_path: str = "master_archetypes.csv") -> pd.DataFrame:
    print("Loading and engineering features …")
    df = load_and_engineer(input_path)

    print("Applying rules-based classification …")
    df = apply_rules(df)

    print("Applying clustering-based classification …")
    df = apply_clustering(df)

    print("Applying modelling-based classification …")
    df = apply_modeling(df)

    print_crosstabs(df)
    print_archetype_profiles(df)

    # ── Save enriched dataset ─────────────────────────────────────────────────
    output_cols = [
        "practicekey", "practicename", "region", "numberofsurgeries",
        "unique_staff_ids", "private_income", "nhs_income_est", "total_income_est",
        "nhs_share", "uda", "nooftreatmentitems",
        "items_per_surgery", "income_per_surgery",
        "has_hygienist", "specialist_flag",
        "archetype_size_rules", "archetype_model_rules", "archetype_rules",
        "cluster_size_id", "cluster_model_id",
        "archetype_size_clust", "archetype_model_clust", "archetype_clust",
        "predicted_income", "income_residual", "predicted_performance_tier",
    ]
    df[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved enriched file → {output_path}  ({len(df):,} rows, {len(output_cols)} columns)")

    return df


if __name__ == "__main__":
    main()
