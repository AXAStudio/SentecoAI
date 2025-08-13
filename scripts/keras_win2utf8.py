import re, unicodedata, numpy as np
from keras.models import load_model
from keras.layers import TextVectorization, StringLookup

from app.utils.ensemble import ProbToLogit

REPL = {
    '\u2019': "'", '\u2018': "'",
    '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-',
    '\u00a0': ' ',
}
RESERVED = { "", "[UNK]", "[PAD]", "[MASK]", "[OOV]",
             "<unk>", "<pad>", "<mask>", "<oov>", "<UNK>", "<PAD>", "<MASK>", "<OOV>" }
INVIS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\u180e\u2060\u2063]")
WS_RE = re.compile(r"\s+")

def norm_token(t: str) -> str:
    if t is None:
        return ""
    # normalize punctuation, strip zero-width/control-ish chars
    t = t.translate(str.maketrans(REPL))
    t = INVIS_RE.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r", "")
    t = WS_RE.sub(" ", t).strip()
    return t

def sanitize_vocab(tokens):
    """Return (cleaned_tokens, kept_indices_in_original)."""
    seen, kept, kept_idx = set(), [], []
    for i, tok in enumerate(tokens):
        tok = norm_token(tok)
        if (tok in RESERVED) or (tok == ""):
            continue
        if tok not in seen:
            seen.add(tok)
            kept.append(tok)
            kept_idx.append(i)
    return kept, kept_idx

def slice_idf_if_needed(layer, kept_idx):
    # Best effort: for tf-idf, first weight typically aligns with vocab length.
    try:
        if getattr(layer, "output_mode", None) == "tf_idf":
            weights = layer.get_weights()
            if weights:
                w0 = weights[0]
                if w0.ndim >= 1 and len(kept_idx) <= w0.shape[0]:
                    weights[0] = w0[kept_idx]
                    layer.set_weights(weights)
                    return True
            # fallback to ones if we can’t slice
            layer.set_vocabulary(layer.get_vocabulary(), idf_weights=np.ones((len(layer.get_vocabulary()),), dtype="float32"))
            return True
    except Exception:
        pass
    return False

def sanitize_model(INPUT_PATH, OUTPUT_PATH, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}

    print("Loading original model...")
    m = load_model(INPUT_PATH, compile=False, custom_objects=custom_objects)

    # Walk all layers (including nested)
    def iter_layers(model):
        stack = list(model.layers)
        while stack:
            l = stack.pop(0)
            yield l
            if hasattr(l, "layers"):
                stack[:0] = list(l.layers)

    changed = 0
    for layer in iter_layers(m):
        try:
            if isinstance(layer, TextVectorization):
                # include special tokens so we can remove them explicitly
                vocab = layer.get_vocabulary(include_special_tokens=True)
                cleaned, kept_idx = sanitize_vocab(vocab)
                # Apply sanitized vocab (tf-idf handled via idf slicing below)
                if getattr(layer, "output_mode", None) == "tf_idf":
                    # set first so shapes line up, then slice idf
                    layer.set_vocabulary(cleaned, idf_weights=np.ones((len(cleaned),), dtype="float32"))
                    slice_idf_if_needed(layer, kept_idx)
                else:
                    layer.set_vocabulary(cleaned)
                changed += 1

            elif isinstance(layer, StringLookup):
                vocab = layer.get_vocabulary(include_special_tokens=True)
                cleaned, _ = sanitize_vocab(vocab)
                # StringLookup should not include OOV/mask in provided vocab
                layer.set_vocabulary(cleaned)
                changed += 1
        except Exception as e:
            print(f"[WARN] Skipped sanitizing {layer.name}: {e}")

    print(f"Sanitized layers: {changed}. Saving…")
    m.save(OUTPUT_PATH)

    print("Verifying clean load…")
    _ = load_model(OUTPUT_PATH, compile=False, custom_objects=custom_objects)

    from app.models.model_selection import load_model_variant
    _ = load_model_variant(path=OUTPUT_PATH)
    print("OK ✓  Saved:", OUTPUT_PATH)



if __name__ == "__main__":
    from app.config import MODEL_VARIANTS

    for variant in MODEL_VARIANTS:
        sanitize_model(
            INPUT_PATH=f"app/models/model_{variant}.keras",
            OUTPUT_PATH=f"app/models/model_{variant}_utf8.keras",
            custom_objects={"ProbToLogit": ProbToLogit} if variant == "ensemble" else None
        )