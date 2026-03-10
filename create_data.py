import pandas as pd
import numpy as np
import random

# 1. Define the full schema from your metadata
columns = [
    "practicekey", "practicecode", "practicename", "brandname", "practiceid", 
    "region", "campus", "postcode", "numberofsurgeries", "numberofchairs", 
    "status", "acquiredon", "nps", "countof_dentistid", "practise_code", 
    "practise_name", "practice_code_and_name", "unique_staff_ids", "clinical_roles", 
    "position_dentist", "position_dental_nurse", "position_receptionist", 
    "position_hygienist", "position_practice_manager", "contractualhours_dentist", 
    "contractualhours_dental_nurse", "contractualhours_receptionist", 
    "contractualhours_hygienist", "contractualhours_practice_manager", 
    "monthkey_x", "active", "nov", "period", "latest_month", "nooftreatmentitems", 
    "nooftreatmentitems_private_standard", "nooftreatmentitems_private_referral", 
    "nooftreatmentitems_nhs_standard", "nooftreatmentitems_nhs_referral", 
    "noofpatients_private_standard", "noofpatients_private_referral", 
    "noofpatients_nhs_standard", "noofpatients_nhs_referral", 
    "chargeprice_private_standard", "chargeprice_private_referral", 
    "chargeprice_nhs_standard", "chargeprice_nhs_referral", 
    "monthkey_y", "privateincome", "uda", "latest_month_y", "countof_snareid"
]

num_rows = 400

# 2. Generate Synthetic Data
data = {}

# Identity & Geography
data["practicekey"] = [f"PK_{1000 + i}" for i in range(num_rows)]
data["practicecode"] = np.random.randint(5000, 9999, size=num_rows)
data["practicename"] = [f"Bupa Dental {random.choice(['North', 'South', 'East', 'West', 'Central'])} {i}" for i in range(num_rows)]
data["brandname"] = ["Bupa Dental Care"] * num_rows
data["practiceid"] = [f"ID-{random.randint(100, 999)}" for _ in range(num_rows)]
data["region"] = [random.choice(["London", "Midlands", "North East", "Scotland", "Wales"]) for _ in range(num_rows)]
data["campus"] = [random.choice(["Campus A", "Campus B", "Main"]) for _ in range(num_rows)]
data["postcode"] = [f"SW{random.randint(1, 20)} {random.randint(1, 9)}AB" for _ in range(num_rows)]

# Capacity & Status
data["numberofsurgeries"] = np.random.randint(2, 10, size=num_rows)
data["numberofchairs"] = data["numberofsurgeries"] + np.random.randint(0, 2, size=num_rows)
data["status"] = ["Active"] * num_rows
data["acquiredon"] = [f"202{random.randint(0, 5)}-01-01" for _ in range(num_rows)]
data["nps"] = np.round(np.random.uniform(20.0, 95.0), 1)

# Staffing (Calculated dynamically)
data["position_dentist"] = np.random.randint(2, 6, size=num_rows)
data["position_dental_nurse"] = data["position_dentist"] + 1
data["position_receptionist"] = np.random.randint(1, 4, size=num_rows)
data["position_hygienist"] = np.random.randint(0, 3, size=num_rows)
data["position_practice_manager"] = [1] * num_rows
data["unique_staff_ids"] = (data["position_dentist"] + data["position_dental_nurse"] + 
                            data["position_receptionist"] + data["position_hygienist"] + 1)

# Hours (Assuming ~37.5 hours per person)
data["contractualhours_dentist"] = data["position_dentist"] * 35.0
data["contractualhours_dental_nurse"] = data["position_dental_nurse"] * 37.5

# Financials & UDAs (Unit of Dental Activity)
data["uda"] = np.random.randint(1000, 8000, size=num_rows)
data["privateincome"] = np.round(np.random.uniform(5000, 50000, size=num_rows), 2)
data["nooftreatmentitems"] = np.random.randint(100, 2000, size=num_rows)

# Fill remaining columns with 0 or NaN to maintain schema
for col in columns:
    if col not in data:
        data[col] = 0

# 3. Create DataFrame and Save
df = pd.DataFrame(data)

# Reorder to match your exact metadata list
df = df[columns]

df.to_csv("master.csv", index=False)

print(f"Created master.csv with {len(df)} rows of practice data.")