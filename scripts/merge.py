import pandas as pd

# === Config ===
file1 = "dataset_deduped.csv"       # first CSV file
file2 = "analyst_ratings_clean.csv"       # second CSV file
output_file = "merged_dataset.csv"

# === Load both datasets ===
df1 = pd.read_csv(file1, encoding="utf-8")
df2 = pd.read_csv(file2, encoding="utf-8")

# === Combine ===
df_combined = pd.concat([df1, df2], ignore_index=True)

# === Save merged dataset ===
df_combined.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Merged dataset saved to {output_file}")
print(f"📊 Final row count: {len(df_combined)}")
