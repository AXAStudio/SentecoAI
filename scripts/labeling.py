# labeling.py
# Run: python labeling.py
# Requires: pip install torch transformers pandas tqdm

import os
import math
import re
import json
import gc
from typing import List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# CONFIG (all inline on purpose)
# -----------------------------
INPUT_CSV  = "analyst_ratings_processed.csv"     # must contain column 'text'
FULL_OUT   = "labeled_full.csv"         # all rows with scores/flags
FINAL_OUT  = "labeled_final.csv"        # clean & high-confidence only
DROPPED_OUT= "dropped_rows.csv"         # junk/uncertain removed

MODEL_ID   = "MoritzLaurer/deberta-v3-large-zeroshot-v1"
BATCH_SIZE = 32               # adjust up/down if VRAM limited
MAX_LEN    = 128               # truncate long headlines safely

# 🔹 Row limit: set to None for all, or an integer like 5000
ROW_LIMIT = 1400000

# Force GPU or stop
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    raise SystemError("❌ No GPU detected! Install PyTorch with CUDA and ensure drivers are set up.")

DTYPE      = torch.float16 if DEVICE == "cuda" else torch.float32

# Confidence gates for keeping rows in FINAL_OUT
MIN_MAX_PROB   = 0.60         # require top-class prob >= 0.60
MIN_MARGIN     = 0.20         # require |pos - neg| >= 0.20

# Simple keyword “guardrails” to fix obvious misses
NEG_TRIGGERS = [
    "downgrade", "lowers price target", "cuts guidance",
    "misses", "missed estimates", "probe", "investigation",
    "lawsuit", "sued", "fine", "penalty", "recall", "outage",
    "breach", "data leak", "delisting", "trading halt", "warning",
    "plunge", "slump", "disappoint", "layoff", "redundancy"
]
POS_TRIGGERS = [
    "upgrade", "raises price target", "beats", "beat estimates",
    "approval", "approved", "clears", "secures permit", "wins contract",
    "record revenue", "record profit", "expands", "acquires",
    "buyback", "repurchase", "dividend increase", "raises guidance",
    "initiates buy", "initiates at buy", "top pick"
]

# Hypotheses for zero-shot NLI
H_POS = "This headline is positive for the company."
H_NEG = "This headline is negative for the company."

# ------------------------------------
# Helpers
# ------------------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s

def is_junk(s: str) -> bool:
    if len(s) < 3:
        return True
    if len(re.findall(r"[^\w\s\$%\-\.]", s)) > 10 and len(s) < 40:
        return True
    return False

def apply_rules(text: str, label: int) -> int:
    t = text.lower()
    if any(k in t for k in NEG_TRIGGERS) and label == 1:
        return -1
    if any(k in t for k in POS_TRIGGERS) and label == -1:
        return 1
    return label

def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: v.to(device) for k, v in obj.items()}
    return obj.to(device)

# ------------------------------------
# Load data
# ------------------------------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Can't find {INPUT_CSV} in: {os.getcwd()}")

df = pd.read_csv(INPUT_CSV, encoding="utf-8")
if "text" not in df.columns:
    raise ValueError("Input CSV must have a 'text' column.")

df["text"] = df["text"].astype(str).map(clean_text)
df = df[~df["text"].map(is_junk)]
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

if ROW_LIMIT is not None and ROW_LIMIT > 0:
    df = df.head(ROW_LIMIT)
    print(f"📏 Limiting to first {len(df)} rows")

print(f"Loaded & cleaned → {len(df)} rows")

# ------------------------------------
# Load model (GPU if available)
# ------------------------------------
print(f"Loading {MODEL_ID} on {DEVICE} …")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(DEVICE, dtype=DTYPE)
model.eval()

if hasattr(model.config, "label2id") and "entailment" in model.config.label2id:
    ENTAIL_IDX = model.config.label2id["entailment"]
else:
    ENTAIL_IDX = 2

@torch.no_grad()
def score_batch(headlines: List[str]) -> Tuple[List[int], List[float], List[float], List[float]]:
    enc_pos = tok(
        headlines,
        [H_POS] * len(headlines),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc_neg = tok(
        headlines,
        [H_NEG] * len(headlines),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc_pos = to_device(enc_pos, DEVICE)
    enc_neg = to_device(enc_neg, DEVICE)

    logits_pos = model(**enc_pos).logits
    logits_neg = model(**enc_neg).logits

    prob_pos = torch.softmax(logits_pos, dim=-1)[:, ENTAIL_IDX]
    prob_neg = torch.softmax(logits_neg, dim=-1)[:, ENTAIL_IDX]

    p_pos = prob_pos.detach().float().cpu().tolist()
    p_neg = prob_neg.detach().float().cpu().tolist()

    labels = []
    margins = []
    for a, b in zip(p_pos, p_neg):
        lbl = 1 if a >= b else -1
        labels.append(lbl)
        margins.append(abs(a - b))
    return labels, p_pos, p_neg, margins

# ------------------------------------
# Run in batches
# ------------------------------------
texts = df["text"].tolist()
out_rows = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Zero-shot labeling"):
    batch = texts[i: i + BATCH_SIZE]
    labels, p_pos, p_neg, margins = score_batch(batch)

    for t, lbl, pp, pn, mg in zip(batch, labels, p_pos, p_neg, margins):
        lbl_fix = apply_rules(t, lbl)
        top_prob = max(pp, pn)
        uncertain = (top_prob < MIN_MAX_PROB) or (mg < MIN_MARGIN)

        out_rows.append({
            "text": t,
            "label": lbl_fix,
            "prob_pos": round(pp, 6),
            "prob_neg": round(pn, 6),
            "margin": round(mg, 6),
            "top_prob": round(top_prob, 6),
            "uncertain": bool(uncertain),
            "rule_flipped": (lbl_fix != lbl)
        })

    if (i // BATCH_SIZE) % 50 == 0:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

full = pd.DataFrame(out_rows)
full.to_csv(FULL_OUT, index=False, encoding="utf-8")
print(f"✅ Saved full outputs → {FULL_OUT}")

final = full[(~full["uncertain"]) & (full["text"].str.len() > 0)].copy()
final.to_csv(FINAL_OUT, index=False, encoding="utf-8")
print(f"✅ Saved final training set → {FINAL_OUT}  (rows: {len(final)})")

dropped = full[full["uncertain"]].copy()
dropped.to_csv(DROPPED_OUT, index=False, encoding="utf-8")
print(f"✅ Saved dropped/uncertain → {DROPPED_OUT}  (rows: {len(dropped)})")

print("\nLabel counts (full set):")
print(full["label"].value_counts())
print("\nLabel counts (final, high-confidence):")
print(final["label"].value_counts())
