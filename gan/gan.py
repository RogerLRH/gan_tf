import os
import pickle

import tensorflow as tf

from gan.function import sample_uniform
from gan.function import sigmoid_loss
from gan.function import GANloader
from gan.function import get_optimizer


class GAN(object):
    def __init__(self, generator, discriminator, data_size, gen_input_len):
        self._generator = generator
        self._discriminator = discriminator
        self._gen_input_len = gen_input_len
        self._opt_params = None

        if isinstance(data_size, int):
            data_size = [data_size]

        # placeholder
        self._reals = tf.placeholder(tf.float32, shape=[None]+data_size)
        self._fakes = tf.placeholder(tf.float32, shape=[None, gen_input_len])

		# nets
        self._G_sample = self._generator(self._fakes)
        self._D_real = self._discriminator(self._reals)
        self._D_fake = self._discriminator(self._G_sample, reuse=True)

		# loss
        self._D_loss = sigmoid_loss(self._D_real) + sigmoid_loss(self._D_fake, one=False)
        self._G_loss = sigmoid_loss(self._D_fake)

        # optimizer
        opt_params = {"learning_rate": 2e-4, "beta1": 0.5}
        self._D_opt = get_optimizer(self._D_loss, opt_type="Adam", opt_params=opt_params, variables=self._discriminator.parameters)
        self._G_opt = get_optimizer(self._G_loss, opt_type="Adam", opt_params=opt_params, variables=self._generator.parameters)

        # saver
        self._saver = tf.train.Saver()
        self._D_saver = tf.train.Saver(self._discriminator.parameters)
        self._G_saver = tf.train.Saver(self._generator.parameters)

    def _feed_dict_D(self, reals, gen_size):
        return {self._reals: reals, self._fakes: sample_uniform(gen_size)}

    def _feed_dict_G(self, gen_size):
        return {self._fakes: sample_uniform(gen_size)}

    def train(self, data_file, sample_dir="samples", ckpt_dir='checkpoints', checkpoint=None, epoches=10000, batch_size=128, D_k=1, G_k=1):
        data = GANloader(data_file, batch_size=batch_size)
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
                    reals = next(data)
                    sess.run(self._D_opt, feed_dict=self._feed_dict_D(reals, gen_size))
                # update G
                for _ in range(G_k):
                    sess.run(self._G_opt, feed_dict=self._feed_dict_G(gen_size))

                if epoch % 100 != 0:
                    continue
                D_loss, G_loss = sess.run([self._D_loss, self._G_loss], feed_dict=self._feed_dict_D(reals, gen_size))
                print('epoch %s: D_loss %.8f; G_loss: %.8f' %(epoch, D_loss, G_loss))

                samples = sess.run(self._G_sample, feed_dict=self._feed_dict_G([16, self._gen_input_len]))
                sample_name = os.path.join(sample_dir, "%s.sample"%epoch)
                pickle.dump(samples, open(sample_name, "wb"))
                ckpt_name = os.path.join(ckpt_dir, "cp_%s"%epoch)
                self._saver.save(sess, ckpt_name)
