# make_ensemble.py
import tensorflow as tf
from tensorflow import keras

# ==== CONFIG ====
MODEL_PATHS = [
    "best_fold_1.keras",
    "best_fold_2.keras",
    "best_fold_3.keras",
    "best_fold_4.keras",
    "best_fold_5.keras",
]
OUT_PATH = "ensemble.keras"
# ================

@keras.utils.register_keras_serializable(package="Custom")
class ProbToLogit(keras.layers.Layer):
    def __init__(self, eps=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg

    def call(self, p):
        p = tf.clip_by_value(p, self.eps, 1.0 - self.eps)
        return tf.math.log(p) - tf.math.log(1.0 - p)

def build_binary_ensemble(model_paths):
    subs = [keras.models.load_model(p) for p in model_paths]
    for m in subs:
        m.trainable = False  # inference-only

    inp = keras.Input(shape=(1,), dtype=tf.string, name="text_input")

    logits = []
    for i, m in enumerate(subs):
        prob = m(inp)                                    # (None, 1) sigmoid prob
        logit = ProbToLogit(name=f"logit_{i+1}")(prob)   # registered layer (serializable)
        logits.append(logit)

    avg_logits = keras.layers.Average(name="avg_logits")(logits)
    out_prob   = keras.layers.Activation("sigmoid", dtype="float32", name="proba")(avg_logits)

    model = keras.Model(inp, out_prob, name="binary_sigmoid_ensemble")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    ens = build_binary_ensemble(MODEL_PATHS)
    ens.save(OUT_PATH)
    print(f"✅ Saved ensemble model to: {OUT_PATH}")
