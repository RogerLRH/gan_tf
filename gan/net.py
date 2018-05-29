import tensorflow as tf
from tensorflow.contrib import layers


class MLPGenerator(object):
    def __init__(self, hidden_sizes, output_size, keep_dropout_p=0.8, name="Generator"):
        self._hidden_sizes = hidden_sizes
        if isinstance(output_size, int):
            output_size = [output_size]
        self._output_size = output_size
        self._keep_dropout_p = keep_dropout_p
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h = inputs
            for hs in self._hidden_sizes[:-1]:
                h = layers.fully_connected(h, hs, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            h = layers.fully_connected(h, self._hidden_sizes[-1], activation_fn=tf.nn.tanh)
            output = tf.reshape(h, [-1]+self._output_size)
        return output

    @property
    def parameters(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MLPDiscriminator(object):
    def __init__(self, hidden_sizes, keep_dropout_p=0.8, name="Discriminator"):
        self._hidden_sizes = hidden_sizes
        self._keep_dropout_p = keep_dropout_p
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h = inputs
            for hs in self._hidden_sizes:
                h = layers.fully_connected(h, hs, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            logit = layers.fully_connected(h, 1, activation_fn=None)
        return logit

    @property
    def parameters(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MnistClassifier(object):
    def __init__(self, keep_dropout_p=0.8, name="Classifier"):
        self._keep_dropout_p = keep_dropout_p
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h = layers.conv2d(inputs, num_outputs=64, kernel_size=5, stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
            h = layers.dropout(h, self._keep_dropout_p)
            h = layers.conv2d(h, num_outputs=128, kernel_size=5, stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
            h = layers.dropout(h, self._keep_dropout_p)
            h = layers.fully_connected(layers.flatten(h), 1024, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
            h = layers.dropout(h, self._keep_dropout_p)
            logits = layers.fully_connected(h, 10, activation_fn=None) # 10 classes
        return logits

    @property
    def parameters(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class ConvGenerator(object):
    def __init__(self, fc_sizes, reshape_size, conv_sizes, keep_dropout_p=0.8, name="Generator"):
        self._fc_sizes = fc_sizes
        self._reshape_size = reshape_size
        self._conv_sizes = conv_sizes
        self._keep_dropout_p = keep_dropout_p
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h = inputs
            for size in self._fc_sizes:
                h = layers.fully_connected(h, size, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            h = tf.reshape(h, [-1]+self._reshape_size)
            for channel, kernel, stride in self._conv_sizes[:-1]:
                h = layers.conv2d_transpose(h, num_outputs=channel, kernel_size=kernel, stride=stride, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            channel, kernel, stride = self._conv_sizes[-1]
            output = layers.conv2d_transpose(h, num_outputs=channel, kernel_size=kernel, stride=stride, activation_fn=tf.nn.tanh)
        return output

    @property
    def parameters(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class ConvDiscriminator(object):
    def __init__(self, conv_sizes, fc_sizes, keep_dropout_p=0.8, name="Discriminator"):
        self._conv_sizes = conv_sizes
        self._fc_sizes = fc_sizes
        self._keep_dropout_p = keep_dropout_p
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h = inputs
            for channel, kernel, stride in self._conv_sizes:
                h = layers.conv2d(h, num_outputs=channel, kernel_size=kernel, stride=stride, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            h = layers.flatten(h)
            for size in self._fc_sizes:
                h = layers.fully_connected(h, size, activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm)
                h = layers.dropout(h, self._keep_dropout_p)
            logits = layers.fully_connected(h, 1, activation_fn=None)
            return logits

    @property
    def parameters(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
