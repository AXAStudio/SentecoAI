import os
import numpy as np
import tensorflow as tf
from typing import Any, List, Union


class TFModel:
    def __init__(self, model: Any, is_ensemble: bool = False):
        self._model = model
        self.is_ensemble = is_ensemble

        # ecide expected input shape once - most text models want [batch],
        # some custom pipelines expect [batch, 1].
        if self.is_ensemble:
            spec = tf.TensorSpec(shape=(None, 1), dtype=tf.string, name="texts")
        else:
            spec = tf.TensorSpec(shape=(None,), dtype=tf.string, name="texts")

        m = self._model  # capture in closure

        @tf.function(input_signature=[spec], reduce_retracing=True)
        def _call(x):
            return m(x, training=False)

        self._call = _call

        # warmup; non-fatal if it fails
        try:
            warm = tf.constant([["warmup"]] if self.is_ensemble else ["warmup"], dtype=tf.string)
            _ = self._call(warm)
        except Exception:
            pass

    def predict(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        t = tf.constant(texts, dtype=tf.string)
        if self.is_ensemble:
            # rank-2 for the signature (None, 1)
            if t.shape.rank == 1:
                t = tf.expand_dims(t, axis=-1)
        else:
            # rank-1 for the signature (None,)
            if t.shape.rank == 2 and t.shape[-1] == 1:
                t = tf.squeeze(t, axis=-1)

        return self._call(t).numpy()
