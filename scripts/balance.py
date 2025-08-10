# balance.py
# Requires: pip install pandas

import pandas as pd

INPUT_CSV = "labeled_full.csv"   # must have 'label' and 'text' columns
OUTPUT_CSV = "balanced.csv"       # output file

# Load
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# Keep only 'label' and 'text'
if not {"label", "text"}.issubset(df.columns):
    raise ValueError("CSV must contain 'label' and 'text' columns.")
df = df[["label", "text"]]

# Keep only -1 and 1
df = df[df["label"].isin([-1, 1])]

# Drop duplicate texts
df = df.drop_duplicates(subset=["text"])

# Balance by undersampling
min_count = df["label"].value_counts().min()
balanced_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .reset_index(drop=True)
)

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ Balanced dataset saved to {OUTPUT_CSV}")
print(balanced_df["label"].value_counts())
