import tensorflow as tf

class Percentage(layers.Layer):

    def __init__(self, epsilon=1E-6):
        super(Percentage, self).__init__(name='percentage')
        self.epsilon = epsilon

    def build(self, input_shape):
        super(Percentage, self).build(input_shape)

    def call(self, x, **kwargs):
        epsilon = tf.constant(self.epsilon)
        x_sum = tf.math.reduce_sum(x, axis=-1, keepdims=True)
        elements = tf.shape(x)[-1]
        x_sum += epsilon * tf.cast(elements, dtype=tf.float32)
        percent = tf.math.divide(x, x_sum, name='percentage')
        percent += epsilon
        return percent

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon': self.epsilon,
        })
        return config
