import tensorflow as tf
from tensorflow.keras.layers import Layer

class GumbelSoftmax(Layer):
    def __init__(self, temperature=1.0, hard=False, **kwargs):
        super(GumbelSoftmax, self).__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, logits):
        noise = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        gumbel_noise = -tf.math.log(-tf.math.log(noise + 1e-20) + 1e-20)
        y = tf.nn.softmax((logits + gumbel_noise) / self.temperature)

        if self.hard:
            y_hard = tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1])
            y = tf.stop_gradient(y_hard - y) + y
        return y
