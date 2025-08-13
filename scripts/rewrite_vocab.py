"""
Rewrite vocab files in Keras models to use UTF-8 encoding and deduplicate tokens.
"""

import io, zipfile, unicodedata, numpy as np
from app.config import MODEL_VARIANTS as variants


# ---- helpers ----
REPLACEMENTS = {
    '\u2019': "'", '\u2018': "'",  # ‘ ’ -> '
    '\u201c': '"', '\u201d': '"',  # “ ” -> "
    '\u2013': '-', '\u2014': '-',  # – — -> -
    '\u00a0': ' ',                 # nbsp -> space
}
DROP_TOKENS = {"", "[UNK]", "[PAD]"}  # should NOT be in vocab files

def norm(s: str) -> str:
    s = s.translate(str.maketrans(REPLACEMENTS))
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def dedupe_keep_order(tokens):
    seen = set()
    kept = []
    kept_idx = []
    for i, t in enumerate(tokens):
        if t in DROP_TOKENS:    # drop reserved/empty
            continue
        if t not in seen:
            seen.add(t)
            kept.append(t)
            kept_idx.append(i)
    return kept, kept_idx

# ---- patcher ----
src = "app/models/model_{}.keras"
dst = "app/models/model_{}_utf8.keras"

for variant in variants:
    with zipfile.ZipFile(src.format(variant), "r") as zin, zipfile.ZipFile(dst.format(variant), "w", zipfile.ZIP_DEFLATED) as zout:
        # First pass: load all assets so we can align any weights if needed.
        assets_text = {}   # filename -> (original_tokens, deduped_tokens, kept_idx)
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.startswith("assets/") and info.filename.lower().endswith(".txt"):
                # Treat all .txt assets as potential vocab files
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("cp1252")
                orig = [norm(line) for line in text.splitlines()]
                # Remove exact duplicates created by normalization (e.g., multiple "-")
                deduped, kept_idx = dedupe_keep_order(orig)
                assets_text[info.filename] = (orig, deduped, kept_idx)

        # Second pass: write files (and adjust any matching IDF weights if present)
        for info in zin.infolist():
            data = zin.read(info.filename)

            # Replace vocab .txt files with deduped content
            if info.filename in assets_text:
                _, deduped, _ = assets_text[info.filename]
                data = ("\n".join(deduped) + "\n").encode("utf-8")

            # If there are TF-IDF weights aligned to a vocab, shrink them using kept indices
            # Heuristic: look for .npy in assets with a name sharing the same layer stem.
            if info.filename.startswith("assets/") and info.filename.lower().endswith(".npy"):
                # Find any vocab file that shares a layer stem with this weights file
                # e.g., "assets/text_vectorization_4_vocabulary.txt" and "assets/text_vectorization_4_idf.npy"
                stem = info.filename.split("assets/")[-1].split(".npy")[0]
                # Try to find a vocab with a common prefix segment
                candidate = None
                for vname in assets_text.keys():
                    vstem = vname.split("assets/")[-1].rsplit(".", 1)[0]
                    if vstem.split("_")[0:3] == stem.split("_")[0:3] or vstem.split("_")[0:2] == stem.split("_")[0:2]:
                        candidate = vname
                        break
                if candidate:
                    orig, deduped, kept_idx = assets_text[candidate]
                    try:
                        arr = np.load(io.BytesIO(data), allow_pickle=False)
                        # If the leading axis equals the vocab length, slice it
                        if arr.shape[0] == len(orig) and len(deduped) < len(orig):
                            arr = arr[kept_idx]
                            buf = io.BytesIO()
                            np.save(buf, arr, allow_pickle=False)
                            data = buf.getvalue()
                    except Exception:
                        pass  # if it's not an IDF array, leave it untouched

            zout.writestr(info, data)

    print("Wrote:", dst.format(variant))
