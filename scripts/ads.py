# remove_non_utf8_and_balance.py
# Run: python remove_non_utf8_and_balance.py
# Requires: pip install pandas

import pandas as pd

# --- Config ---
INPUT_CSV  = "balanced.csv"   # must contain columns 'text' and 'label'
OUTPUT_CSV = "balanced_utf8.csv"   # clean + balanced output
SEED       = 42

def is_utf8(s: str) -> bool:
    try:
        s.encode("utf-8", "strict")
        return True
    except UnicodeEncodeError:
        return False

# --- Load (no errors param) ---
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# Keep only needed columns
df = df[["text", "label"]].copy()

# Ensure proper types
df["text"] = df["text"].astype(str)
df = df[df["label"].isin([-1, 1])]

# Remove rows not valid UTF-8
mask = df["text"].map(is_utf8)
df = df[mask].reset_index(drop=True)

print(f"✅ Rows after UTF-8 filter: {len(df)}")

# --- Balance by undersampling ---
counts = df["label"].value_counts()
min_count = counts.min()

balanced = (
    df.groupby("label", group_keys=False)
      .apply(lambda g: g.sample(min_count, random_state=SEED))
      .reset_index(drop=True)
      .sample(frac=1.0, random_state=SEED)
)

# --- Save ---
balanced.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

print(f"📦 Saved balanced UTF-8 dataset → {OUTPUT_CSV}")
print("Label counts (balanced):")
print(balanced["label"].value_counts())
