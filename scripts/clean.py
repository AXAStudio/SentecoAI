import pandas as pd
import re

INPUT  = "final_data.csv"
OUTPUT = "final_data_clean.csv"
TEXT   = "text"

def encodable_cp1252(s: str) -> bool:
    try:
        s.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False

def has_control_chars(s: str) -> bool:
    # disallow ASCII control chars except \t \n \r
    return bool(re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", s))

df = pd.read_csv(INPUT, encoding="utf-8")

# Ensure required columns exist
if TEXT not in df.columns or "label" not in df.columns:
    raise ValueError(f"CSV must have '{TEXT}' and 'label' columns.")

# Basic text cleanup
df[TEXT] = df[TEXT].astype(str).str.strip()

# Build mask of “bad” rows
bad_mask = (
    df[TEXT].isna() |
    (df[TEXT] == "") |
    (df[TEXT].str.len() < 3) |
    (df[TEXT].str.contains("\ufffd")) |
    (df[TEXT].apply(has_control_chars)) |
    (~df[TEXT].apply(encodable_cp1252)) |
    (~df["label"].isin([-1, 1]))
)

removed = int(bad_mask.sum())
df_clean = df.loc[~bad_mask].copy()

# Optional: drop duplicate texts
before_dupes = len(df_clean)
df_clean.drop_duplicates(subset=[TEXT], keep="first", inplace=True)
dupe_removed = before_dupes - len(df_clean)

# Save
df_clean.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"✅ Saved cleaned dataset to {OUTPUT}")
print(f"🧹 Removed {removed} bad rows, {dupe_removed} duplicates")
print("📊 Label balance after cleaning:")
print(df_clean["label"].value_counts())
