import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

# Load dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/cleaned_composition.csv")

print("Original shape:", df.shape)

# Initialize featurizer
featurizer = ElementProperty.from_preset("magpie")

# Store features
features = []
targets = []

for i, row in df.iterrows():
    formula = row["clean_formula"]
    zt = row["ZT"]

    try:
        comp = Composition(formula)

        feat = featurizer.featurize(comp)

        features.append(feat)
        targets.append(zt)

    except:
        continue  # skip invalid formulas

# Convert to DataFrame
feature_names = featurizer.feature_labels()
X = pd.DataFrame(features, columns=feature_names)
y = pd.Series(targets, name="ZT")

# Combine
final_df = pd.concat([X, y], axis=1)

print("Final dataset shape:", final_df.shape)

# Save dataset
final_df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_ml_dataset.csv", index=False)

print("✅ Feature extraction complete!")

