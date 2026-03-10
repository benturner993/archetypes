import pandas as pd

# Full schema extracted from the metadata image
columns_and_types = {
    # Practice Identity
    "practicekey": "str",
    "practicecode": "Int64",
    "practicename": "str",
    "brandname": "str",
    "practiceid": "str",
    "region": "str",
    "campus": "str",
    "postcode": "str",
    "numberofsurgeries": "Int64",
    "numberofchairs": "Int64",
    "status": "str",
    "acquiredon": "str",
    "nps": "float64",
    "countof_dentistid": "float64",
    
    # Practice Details & Staffing
    "practise_code": "float64",
    "practise_name": "str",
    "practice_code_and_name": "str",
    "unique_staff_ids": "float64",
    "clinical_roles": "float64",
    "position_dentist": "float64",
    "position_dental_nurse": "float64",
    "position_receptionist": "float64",
    "position_hygienist": "float64",
    "position_practice_manager": "float64",
    
    # Contractual Hours
    "contractualhours_dentist": "float64",
    "contractualhours_dental_nurse": "float64",
    "contractualhours_receptionist": "float64",
    "contractualhours_hygienist": "float64",
    "contractualhours_practice_manager": "float64",
    
    # Activity & Performance Metrics
    "monthkey_x": "float64",
    "active": "Int64",
    "nov": "Int64",
    "period": "Int64",
    "latest_month": "Int64",
    "nooftreatmentitems": "Int64",
    "nooftreatmentitems_private_standard": "float64",
    "nooftreatmentitems_private_referral": "float64",
    "nooftreatmentitems_nhs_standard": "float64",
    "nooftreatmentitems_nhs_referral": "float64",
    
    # Patient & Income Data
    "noofpatients_private_standard": "float64",
    "noofpatients_private_referral": "float64",
    "noofpatients_nhs_standard": "float64",
    "noofpatients_nhs_referral": "float64",
    "chargeprice_private_standard": "float64",
    "chargeprice_private_referral": "float64",
    "chargeprice_nhs_standard": "float64",
    "chargeprice_nhs_referral": "float64",
    "monthkey_y": "float64",
    "privateincome": "float64",
    "uda": "float64",
    "latest_month_y": "float64",
    "countof_snareid": "float64"
}

# Create the empty DataFrame with the specified types
df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in columns_and_types.items()})

# Export to CSV
df.to_csv("metadata.csv", index=False)

print(f"Created metadata.csv with {len(df.columns)} columns.")