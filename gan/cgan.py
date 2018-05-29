import os
import pickle

import tensorflow as tf

from gan.function import sample_uniform
from gan.function import sample_class
from gan.function import sigmoid_loss
from gan.function import GANClassloader
from gan.function import get_optimizer


def concat_inputs_labels(inputs, labels):
    input_size = inputs.shape[1:]
    if len(input_size) == 1:
        return tf.concat((inputs, labels), 1)
    num_class = labels.shape[1]
    reshape_size = [-1] + [1 for _ in range(len(input_size) - 1)] + [num_class]
    labels = tf.reshape(labels, reshape_size)
    labels *= tf.ones([1] + [i for i in input_size[:-1]] + [num_class])
    return tf.concat((inputs, labels), -1)


class CGAN(object):
    def __init__(self, generator, discriminator, data_size, num_class, gen_input_len):
        self._generator = generator
        self._discriminator = discriminator
        if isinstance(data_size, int):
            data_size = [data_size]
        self._data_size = data_size
        self._num_class = num_class
        self._gen_input_len = gen_input_len

        # placeholder
        self._reals = tf.placeholder(tf.float32, shape=[None]+data_size)
        self._labels = tf.placeholder(tf.float32, shape=[None, num_class])
        self._fakes = tf.placeholder(tf.float32, shape=[None, gen_input_len])

		# nets
        self._G_sample = self._generator(tf.concat([self._fakes, self._labels], 1))
        self._D_real = self._discriminator(concat_inputs_labels(self._reals, self._labels))
        self._D_fake = self._discriminator(concat_inputs_labels(self._G_sample, self._labels), reuse=True)

		# loss
        self._D_loss = sigmoid_loss(self._D_real) + sigmoid_loss(self._D_fake, one=False)
        self._G_loss = sigmoid_loss(self._D_fake)

        # optimizer
        opt_params = {"learning_rate": 2e-4, "beta1": 0.5}
        self._D_opt = get_optimizer(self._D_loss, opt_type="Adam", opt_params=opt_params, variables=self._discriminator.parameters)
        self._G_opt = get_optimizer(self._G_loss, opt_type="Adam", opt_params=opt_params, variables=self._generator.parameters)
        opt_params = {"learning_rate": 1e-4}

        # saver
        self._saver = tf.train.Saver()
        self._D_saver = tf.train.Saver(self._discriminator.parameters)
        self._G_saver = tf.train.Saver(self._generator.parameters)

    def _feed_dict_D(self, reals, labels, gen_size):
        fakes = sample_uniform(gen_size)
        return {self._reals: reals, self._labels: labels, self._fakes: fakes}

    def _feed_dict_G(self, gen_size):
        labels = sample_class(gen_size[0], self._num_class)
        fakes = sample_uniform(gen_size)
        return {self._labels: labels, self._fakes: fakes}

    def train(self, data_file, sample_dir="samples", ckpt_dir='checkpoints', checkpoint=None, epoches=10000, batch_size=128, D_k=1, G_k=1):
        data = GANClassloader(data_file, batch_size=batch_size)
        gen_size = [batch_size, self._gen_input_len]
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if checkpoint:
                self._saver.restore(sess, checkpoint)

            for epoch in range(epoches):
                # update D
                for _ in range(D_k):
                    reals, labels = next(data)
                    sess.run(self._D_opt, feed_dict=self._feed_dict_D(reals, labels, gen_size))
                # update G
                for _ in range(G_k):
                    sess.run(self._G_opt, feed_dict=self._feed_dict_G(gen_size))

                if epoch % 100 != 0:
                    continue
                D_loss, G_loss = sess.run([self._D_loss, self._G_loss], feed_dict=self._feed_dict_D(reals, labels, gen_size))
                print('epoch %s: D_loss %.8f; G_loss: %.8f' %(epoch, D_loss, G_loss))

                feed_dict_G = self._feed_dict_G([16, self._gen_input_len])
                samples = sess.run(self._G_sample, feed_dict=feed_dict_G)
                sample_name = os.path.join(sample_dir, "%s.sample"%epoch)
                ckpt_name = os.path.join(ckpt_dir, "cp_%s"%epoch)
                pickle.dump((feed_dict_G[self._labels], samples), open(sample_name, "wb"))
                self._saver.save(sess, ckpt_name)
