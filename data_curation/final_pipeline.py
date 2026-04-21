import pandas as pd
import numpy as np

# Load latest dataset (after feature extraction + NaN cleaning)
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_45_features_dataset_clean.csv")

print("Initial shape:", df.shape)

# -------------------------------
# 1. REMOVE DUPLICATES
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# 2. REMOVE NEGATIVE ZT
# -------------------------------
df = df[df["ZT"] >= 0]

# -------------------------------
# 3. FIX EXTREME VALUES (Z FEATURES)
# -------------------------------
problem_cols = ["avg_Z", "max_Z", "min_Z", "range_Z", "std_Z", "Z_diff"]

for col in problem_cols:
    if col in df.columns:
        df[col] = df[col].clip(0, 200)

# -------------------------------
# 4. FINAL SAFETY CLIP (ALL NUMERIC)
# -------------------------------
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].clip(-1e6, 1e6)

# -------------------------------
# FINAL CHECK
# -------------------------------
print("Final shape:", df.shape)
print("NaNs:", df.isnull().sum().sum())

# -------------------------------
# SAVE FINAL DATASET
# -------------------------------
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_ready_dataset.csv", index=False)

print("\n🔥 FINAL DATASET READY FOR ML 🔥")
