# make_ensemble.py
import tensorflow as tf
from tensorflow import keras


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

