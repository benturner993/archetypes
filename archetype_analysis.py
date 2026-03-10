"""
archetype_analysis.py
---------------------
Assigns every practice in master.csv to a position in the Archetypes Progress
framework: a 4x4 matrix of Size x Model.

Size bands   : Small/Foundation | Medium/Core | Large/Advanced | Flagship
Model bands  : NHS Led | Balanced Mixed | Private Led Mixed | Specialist/Referral Hub

Three labelling strategies are implemented:
  1. apply_rules()    - deterministic heuristics derived from the framework
  2. apply_clustering() - unsupervised K-Means (size and model independently)
  3. apply_scoring()  - composite affinity scoring (0-100 per archetype)

N/A zone enforcement (per framework slide):
  NHS Led practices cannot be Large/Advanced or Flagship -- they are
  reclassified to Balanced Mixed if the size rule fires at those levels.

Data notes (synthetic master.csv):
  - nooftreatmentitems_nhs_standard / _referral columns are all zero.
    NHS activity is proxied via UDA counts x 28 (standard UDA rate).
  - countof_snareid and chargeprice_private_referral are all zero.
    Specialist/Referral Hub is proxied via hygienist presence + private
    income intensity.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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


# ==============================================================================
# 3 · COMPOSITE ARCHETYPE SCORING
# ==============================================================================

# Affinity weights per model archetype.
# Each key must match a component column name computed in apply_scoring().
# Weights within each archetype must sum to 1.0.
AFFINITY_WEIGHTS = {
    "NHS Led": {
        "nhs_share_score":       0.50,
        "uda_score":             0.30,
        "anti_specialist_score": 0.20,
    },
    "Balanced Mixed": {
        "balance_score":           0.60,
        "anti_specialist_score":   0.20,
        "private_intensity_score": 0.10,
        "uda_score":               0.10,
    },
    "Private Led Mixed": {
        "private_share_score":     0.45,
        "private_intensity_score": 0.35,
        "anti_specialist_score":   0.20,
    },
    "Specialist/Referral Hub": {
        "specialist_score":        0.40,
        "private_intensity_score": 0.35,
        "private_share_score":     0.25,
    },
}

# Size index weights (components ranked by percentile, then combined)
SIZE_INDEX_WEIGHTS = {
    "numberofsurgeries":        0.40,
    "unique_staff_ids":         0.40,
    "contractualhours_dentist": 0.20,
}

LOW_CONFIDENCE_THRESHOLD = 10  # gap < 10 points = blend/boundary practice


def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank scaled to 0-100; NaN assigned neutral 50."""
    return series.rank(pct=True).fillna(0.5) * 100


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a 0-100 affinity score to each practice for each model archetype.
    The highest-scoring archetype wins. Size is determined by a weighted
    composite size index cut at portfolio quartiles.

    Adds columns:
      size_index               - Weighted percentile-rank size index (0-100)
      archetype_size_score     - Size band from size index quartile cut
      affinity_<archetype>     - 0-100 affinity score for each model archetype
                                 (column name sanitised: spaces/slashes -> _)
      archetype_model_score    - Model band (highest affinity, NA zones enforced)
      affinity_primary         - Highest affinity score
      affinity_secondary       - Second-highest affinity score
      affinity_confidence_gap  - Primary minus secondary (higher = clearer fit)
      archetype_blend          - True if confidence gap < LOW_CONFIDENCE_THRESHOLD
      archetype_score          - Combined 'Size | Model' label
    """
    df = df.copy()

    # ── Signal components (0-100 percentile scale) ────────────────────────────
    df["nhs_share_score"]         = _pct_rank(df["nhs_share"])
    df["private_share_score"]     = _pct_rank(1 - df["nhs_share"].fillna(0))
    df["private_intensity_score"] = _pct_rank(df["private_income_per_chair"])
    df["uda_score"]               = _pct_rank(df["uda"])
    df["specialist_score"]        = _pct_rank(df["specialist_flag"].astype(float) * 0.5
                                              + df["private_income_per_chair"].rank(pct=True).fillna(0) * 0.5)
    df["anti_specialist_score"]   = 100 - df["specialist_score"]
    df["balance_score"]           = (
        1 - (df["nhs_share"].fillna(0.5) - 0.5).abs() / 0.5
    ) * 100

    # ── Affinity scores ───────────────────────────────────────────────────────
    affinity_matrix = pd.DataFrame(
        {
            arch: sum(df[comp] * weight for comp, weight in weights.items())
            for arch, weights in AFFINITY_WEIGHTS.items()
        },
        index=df.index,
    )

    # Sanitised column names for output
    for arch in MODEL_LABELS:
        col = "affinity_" + arch.lower().replace("/", "_").replace(" ", "_")
        df[col] = affinity_matrix[arch].round(1)

    df["archetype_model_score"]   = affinity_matrix.idxmax(axis=1)
    df["affinity_primary"]        = affinity_matrix.max(axis=1).round(1)
    df["affinity_secondary"]      = affinity_matrix.apply(
        lambda row: row.nlargest(2).iloc[1], axis=1
    ).round(1)
    df["affinity_confidence_gap"] = (df["affinity_primary"] - df["affinity_secondary"]).round(1)
    df["archetype_blend"]         = df["affinity_confidence_gap"] < LOW_CONFIDENCE_THRESHOLD

    # ── Size index ────────────────────────────────────────────────────────────
    df["size_index"] = sum(
        _pct_rank(df[col]) * weight for col, weight in SIZE_INDEX_WEIGHTS.items()
    )
    q25, q50, q75 = df["size_index"].quantile([0.25, 0.50, 0.75])
    df["archetype_size_score"] = pd.cut(
        df["size_index"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=SIZE_LABELS,
    )

    # ── N/A zone enforcement ──────────────────────────────────────────────────
    df["archetype_model_score"] = enforce_na_zones(
        df["archetype_size_score"].astype(str), df["archetype_model_score"]
    )
    df["archetype_score"] = df["archetype_size_score"].astype(str) + " | " + df["archetype_model_score"]

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("COMPOSITE SCORING -- Model band distribution")
    print("-" * 60)
    print(df["archetype_model_score"].value_counts().reindex(MODEL_LABELS).to_string())
    print("\n" + "-" * 60)
    print("COMPOSITE SCORING -- Confidence gap summary")
    print("-" * 60)
    print(df["affinity_confidence_gap"].describe().round(1).to_string())
    n_blend = df["archetype_blend"].sum()
    print(f"\nBlend practices (gap < {LOW_CONFIDENCE_THRESHOLD}): {n_blend} ({n_blend/len(df)*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4 · CROSSTAB OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def print_crosstabs(df: pd.DataFrame) -> None:
    """Print 16-cell crosstabs for all three labelling approaches."""

    size_order  = SIZE_LABELS
    model_order = MODEL_LABELS

    approaches = [
        ("Rules-Based",   "archetype_size_rules",        "archetype_model_rules"),
        ("Clustering",    "archetype_size_clust",         "archetype_model_clust"),
        ("Scoring",       "archetype_size_score",         "archetype_model_score"),
    ]

    for label, size_col, model_col in approaches:
        print("\n" + "-" * 70)
        print(f"PRACTICE COUNT CROSSTAB -- {label} Approach")
        print("  Rows = Size  |  Columns = Model")
        print("-" * 70)
        ct = pd.crosstab(
            df[size_col].astype(str),
            df[model_col],
            margins=True,
            margins_name="TOTAL",
        ).reindex(index=size_order + ["TOTAL"], columns=model_order + ["TOTAL"], fill_value=0)
        print(ct.to_string())

    # Blend flag breakdown
    print("\n" + "-" * 70)
    print("BLEND PRACTICES (low confidence gap) BY SCORING ARCHETYPE")
    print("-" * 70)
    blend_ct = pd.crosstab(
        df["archetype_score"],
        df["archetype_blend"].map({True: "Blend", False: "Clear"}),
        margins=True,
        margins_name="TOTAL",
    )
    print(blend_ct.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 5 · ARCHETYPE PROFILE SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def print_archetype_profiles(df: pd.DataFrame) -> None:
    """Summarise key metrics for each rules-based and scoring size x model cell."""
    for approach_label, size_col, model_col in [
        ("Rules-Based", "archetype_size_rules",  "archetype_model_rules"),
        ("Scoring",     "archetype_size_score",   "archetype_model_score"),
    ]:
        print("\n" + "-" * 70)
        print(f"ARCHETYPE PROFILES -- Mean key metrics ({approach_label})")
        print("-" * 70)
        profile = (
            df.groupby([size_col, model_col], observed=True)
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

    print("Applying composite scoring classification …")
    df = apply_scoring(df)

    print_crosstabs(df)
    print_archetype_profiles(df)

    # ── Save enriched dataset ─────────────────────────────────────────────────
    affinity_cols = [
        c for c in df.columns if c.startswith("affinity_")
        and c not in ("affinity_primary", "affinity_secondary", "affinity_confidence_gap")
    ]
    output_cols = [
        "practicekey", "practicename", "region", "numberofsurgeries",
        "unique_staff_ids", "private_income", "nhs_income_est", "total_income_est",
        "nhs_share", "uda", "nooftreatmentitems",
        "items_per_surgery", "income_per_surgery",
        "has_hygienist", "specialist_flag",
        "archetype_size_rules", "archetype_model_rules", "archetype_rules",
        "cluster_size_id", "cluster_model_id",
        "archetype_size_clust", "archetype_model_clust", "archetype_clust",
        "size_index", "archetype_size_score", "archetype_model_score", "archetype_score",
        "affinity_primary", "affinity_secondary", "affinity_confidence_gap", "archetype_blend",
    ] + affinity_cols
    df[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved enriched file → {output_path}  ({len(df):,} rows, {len(output_cols)} columns)")

    return df


if __name__ == "__main__":
    main()
