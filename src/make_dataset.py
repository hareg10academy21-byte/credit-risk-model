import pandas as pd
import os

# =========================
# LOAD RAW DATA
# =========================
df = pd.read_csv("data/data.csv")

print("Raw data shape:", df.shape)

# =========================
# CREATE TARGET (TASK 4 LOGIC RESTORED)
# =========================
# FraudResult: 1 = risky, 0 = safe
df["is_high_risk"] = df["FraudResult"]

# =========================
# OPTIONAL CLEANING
# =========================
df = df.drop(columns=["FraudResult"])

# handle missing values
df = df.fillna(0)

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("data/processed", exist_ok=True)

# =========================
# SAVE PROCESSED DATA
# =========================
output_path = "data/processed/processed.csv"
df.to_csv(output_path, index=False)

print("Processed dataset saved at:", output_path)
print("Columns:", df.columns.tolist())