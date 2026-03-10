"""
create_data.py
--------------
Generates a realistic synthetic master.csv of 400 dental practices.

Design principles:
  - Each practice is assigned a latent 'practice_type' that drives correlated
    signals across income, staffing, patients and referral activity.
  - Practice types: Generalist (40%), Mixed (40%), Specialist (20%)
  - Specialist practices genuinely receive referral patients and have SNARE
    registrations — allowing proper referral-based specialist classification.
  - All patient, treatment-item, income and charge columns are populated and
    internally consistent (items ≈ patients × items-per-patient, etc.).
"""

import pandas as pd
import numpy as np
import random

rng = np.random.default_rng(seed=42)   # reproducible

columns = [
    "practicekey", "practicecode", "practicename", "brandname", "practiceid",
    "region", "campus", "postcode", "numberofsurgeries", "numberofchairs",
    "status", "acquiredon", "nps", "countof_dentistid", "practise_code",
    "practise_name", "practice_code_and_name", "unique_staff_ids", "clinical_roles",
    "position_dentist", "position_dental_nurse", "position_receptionist",
    "position_hygienist", "position_practice_manager",
    "contractualhours_dentist", "contractualhours_dental_nurse",
    "contractualhours_receptionist", "contractualhours_hygienist",
    "contractualhours_practice_manager",
    "monthkey_x", "active", "nov", "period", "latest_month",
    "nooftreatmentitems",
    "nooftreatmentitems_private_standard", "nooftreatmentitems_private_referral",
    "nooftreatmentitems_nhs_standard", "nooftreatmentitems_nhs_referral",
    "noofpatients_private_standard", "noofpatients_private_referral",
    "noofpatients_nhs_standard", "noofpatients_nhs_referral",
    "chargeprice_private_standard", "chargeprice_private_referral",
    "chargeprice_nhs_standard", "chargeprice_nhs_referral",
    "monthkey_y", "privateincome", "nhsincome", "latest_month_y", "countof_snareid",
]

N = 400
data = {}

# ── Identity & Geography ──────────────────────────────────────────────────────
regions = ["London", "Midlands", "North East", "Scotland", "Wales"]
data["practicekey"]   = [f"PK_{1000 + i}" for i in range(N)]
data["practicecode"]  = rng.integers(5000, 9999, size=N).tolist()
data["practicename"]  = [
    f"Bupa Dental {random.choice(['North','South','East','West','Central'])} {i}"
    for i in range(N)
]
data["brandname"]  = ["Bupa Dental Care"] * N
data["practiceid"] = [f"ID-{rng.integers(100, 999)}" for _ in range(N)]
data["region"]     = [random.choice(regions) for _ in range(N)]
data["campus"]     = [random.choice(["Campus A", "Campus B", "Main"]) for _ in range(N)]
data["postcode"]   = [f"SW{rng.integers(1,20)} {rng.integers(1,9)}AB" for _ in range(N)]

# ── Physical capacity ─────────────────────────────────────────────────────────
surgeries = rng.integers(2, 10, size=N)
data["numberofsurgeries"] = surgeries
data["numberofchairs"]    = (surgeries + rng.integers(0, 2, size=N)).tolist()
data["status"]            = ["Active"] * N
data["acquiredon"]        = [f"202{rng.integers(0,5)}-01-01" for _ in range(N)]
data["nps"]               = np.round(rng.uniform(20.0, 95.0, size=N), 1).tolist()

# ── Staffing ──────────────────────────────────────────────────────────────────
n_dentists   = rng.integers(2, 6, size=N)
n_nurses     = n_dentists + 1
n_reception  = rng.integers(1, 4, size=N)
n_hygienists = rng.integers(0, 3, size=N)
n_pm         = np.ones(N, dtype=int)

data["position_dentist"]          = n_dentists.tolist()
data["position_dental_nurse"]     = n_nurses.tolist()
data["position_receptionist"]     = n_reception.tolist()
data["position_hygienist"]        = n_hygienists.tolist()
data["position_practice_manager"] = n_pm.tolist()
data["unique_staff_ids"]          = (n_dentists + n_nurses + n_reception + n_hygienists + 1).tolist()
data["clinical_roles"]            = (n_dentists + n_nurses + n_hygienists).tolist()
data["countof_dentistid"]         = n_dentists.tolist()

# Contracted hours
data["contractualhours_dentist"]          = (n_dentists * 35.0).tolist()
data["contractualhours_dental_nurse"]     = (n_nurses   * 37.5).tolist()
data["contractualhours_receptionist"]     = (n_reception * 37.5).tolist()
data["contractualhours_hygienist"]        = (n_hygienists * 30.0).tolist()
data["contractualhours_practice_manager"] = (n_pm * 37.5).tolist()

# ── Practice type (latent variable driving correlated signals) ────────────────
# Specialist:  ~20% — receive referral patients, high private income
# Mixed:       ~40% — moderate referral activity, balanced mix
# Generalist:  ~40% — little/no referral activity, NHS-heavy
#
# Specialist more likely in larger practices with hygienists.
orientation = rng.beta(2, 3, size=N)                          # 0=NHS-heavy, 1=private-heavy
orientation += np.clip((surgeries - surgeries.mean()) * 0.03, -0.1, 0.15)
orientation += np.clip(n_hygienists * 0.05, 0, 0.15)
orientation = np.clip(orientation, 0, 1)

is_specialist = orientation > 0.72           # ~20%
is_mixed      = (orientation > 0.40) & ~is_specialist   # ~40%
is_generalist = ~(is_specialist | is_mixed)  # ~40%

size_mult = surgeries / surgeries.mean()     # scale income with physical footprint

# ── Income ────────────────────────────────────────────────────────────────────
# nhsincome: generalists have more NHS contract income; specialists less
nhs_base = np.where(
    is_specialist, rng.uniform(15_000,  70_000, N),
    np.where(is_mixed, rng.uniform(50_000, 150_000, N),
                        rng.uniform(80_000, 230_000, N))
)
data["nhsincome"]     = np.round(nhs_base * size_mult, 2).tolist()

priv_base = np.where(
    is_specialist, rng.uniform(120_000, 380_000, N),
    np.where(is_mixed, rng.uniform(35_000, 160_000, N),
                        rng.uniform(8_000,  65_000, N))
)
data["privateincome"] = np.round(priv_base * size_mult, 2).tolist()

# ── Referral patients ─────────────────────────────────────────────────────────
# noofpatients_private_referral: patients referred IN for specialist private work
priv_ref_pts = np.where(
    is_specialist, rng.integers(25, 90,  N),
    np.where(is_mixed, rng.integers(5,  30,  N),
                        rng.integers(0,  10,  N))
).astype(int)

# noofpatients_nhs_referral: patients referred IN for NHS specialist care
nhs_ref_pts = np.where(
    is_specialist, rng.integers(10, 50,  N),
    np.where(is_mixed, rng.integers(2,  15,  N),
                        rng.integers(0,   5,  N))
).astype(int)

data["noofpatients_private_referral"] = priv_ref_pts.tolist()
data["noofpatients_nhs_referral"]     = nhs_ref_pts.tolist()

# Standard (non-referral) patients — correlated with practice type
priv_std_pts = np.where(
    is_specialist, rng.integers(80,  260, N),
    np.where(is_mixed, rng.integers(100, 420, N),
                        rng.integers(50,  210, N))
).astype(int)

nhs_std_pts = np.where(
    is_specialist, rng.integers(60,  200, N),
    np.where(is_mixed, rng.integers(150, 500, N),
                        rng.integers(200, 620, N))
).astype(int)

data["noofpatients_private_standard"] = priv_std_pts.tolist()
data["noofpatients_nhs_standard"]     = nhs_std_pts.tolist()

# ── Treatment items ───────────────────────────────────────────────────────────
# Items per patient vary by type; referral cases are higher-complexity
ipp_priv_std  = rng.uniform(3.5, 6.5, N)
ipp_nhs_std   = rng.uniform(4.0, 7.0, N)
ipp_priv_ref  = rng.uniform(3.0, 6.0, N)   # specialist procedures
ipp_nhs_ref   = rng.uniform(2.0, 4.0, N)

ti_priv_std  = np.round(priv_std_pts * ipp_priv_std).astype(int)
ti_nhs_std   = np.round(nhs_std_pts  * ipp_nhs_std ).astype(int)
ti_priv_ref  = np.round(priv_ref_pts * ipp_priv_ref).astype(int)
ti_nhs_ref   = np.round(nhs_ref_pts  * ipp_nhs_ref ).astype(int)

data["nooftreatmentitems_private_standard"] = ti_priv_std.tolist()
data["nooftreatmentitems_nhs_standard"]     = ti_nhs_std.tolist()
data["nooftreatmentitems_private_referral"] = ti_priv_ref.tolist()
data["nooftreatmentitems_nhs_referral"]     = ti_nhs_ref.tolist()
data["nooftreatmentitems"]                  = (ti_priv_std + ti_nhs_std +
                                               ti_priv_ref + ti_nhs_ref).tolist()

# ── Charge prices (average item price, £) ─────────────────────────────────────
# Private standard: check-ups, fillings, hygiene — £80–£250
cp_priv_std = np.where(priv_std_pts > 0, np.round(rng.uniform(80,  250, N), 2), 0.0)
# NHS standard: banded NHS charges — £25–£80
cp_nhs_std  = np.where(nhs_std_pts  > 0, np.round(rng.uniform(25,   80, N), 2), 0.0)
# Private referral: specialist procedures — £300–£1400
cp_priv_ref = np.where(priv_ref_pts > 0, np.round(rng.uniform(300, 1400, N), 2), 0.0)
# NHS referral: specialist NHS — £100–£350
cp_nhs_ref  = np.where(nhs_ref_pts  > 0, np.round(rng.uniform(100,  350, N), 2), 0.0)

data["chargeprice_private_standard"] = cp_priv_std.tolist()
data["chargeprice_nhs_standard"]     = cp_nhs_std.tolist()
data["chargeprice_private_referral"] = cp_priv_ref.tolist()
data["chargeprice_nhs_referral"]     = cp_nhs_ref.tolist()

# ── SNARE: specialist network registrations ────────────────────────────────────
# countof_snareid = number of active specialist registrations for the practice
snare = np.where(
    is_specialist, rng.integers(5, 30, N),
    np.where(is_mixed, rng.integers(0,  8, N),
                        rng.integers(0,  3, N))
).astype(int)
data["countof_snareid"] = snare.tolist()

# ── Admin / housekeeping columns ──────────────────────────────────────────────
data["practise_code"]           = data["practicecode"]
data["practise_name"]           = data["practicename"]
data["practice_code_and_name"]  = [
    f"{c} – {n}" for c, n in zip(data["practicecode"], data["practicename"])
]
data["monthkey_x"]   = ["2024-11"] * N
data["active"]       = [1] * N
data["nov"]          = [1] * N
data["period"]       = ["2024-11"] * N
data["latest_month"] = ["2024-11"] * N
data["monthkey_y"]   = ["2024-11"] * N
data["latest_month_y"] = ["2024-11"] * N

# ── Assemble & save ────────────────────────────────────────────────────────────
df = pd.DataFrame(data)[columns]
df.to_csv("master.csv", index=False)

print(f"Created master.csv with {len(df)} rows.")
print()

# Sanity check
specialist_flag = (
    (df["noofpatients_private_referral"] + df["noofpatients_nhs_referral"])
    >= (df["noofpatients_private_referral"] + df["noofpatients_nhs_referral"]).quantile(0.75)
)
print(f"Practice type breakdown (orientation-based):")
print(f"  Specialist : {is_specialist.sum()} ({is_specialist.mean()*100:.0f}%)")
print(f"  Mixed      : {is_mixed.sum()} ({is_mixed.mean()*100:.0f}%)")
print(f"  Generalist : {is_generalist.sum()} ({is_generalist.mean()*100:.0f}%)")
print()
print("Referral patient summary:")
print(df[["noofpatients_private_referral","noofpatients_nhs_referral","countof_snareid"]].describe().round(1).to_string())
print()
print("Income summary:")
print(df[["nhsincome","privateincome"]].describe().round(0).to_string())
