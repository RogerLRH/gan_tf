import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.image as mpimg


def sample_uniform(size):
    return np.random.uniform(-1., 1., size)


def sample_class(batch_size, num_class):
    labels = np.random.randint(num_class, size=batch_size)
    return np.eye(num_class)[labels]


def sigmoid_loss(logits, one=True):
    if one:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def calcul_accuracy(logits, labels):
    logits = tf.argmax(logits, axis=1)
    labels = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.to_float(tf.equal(logits, labels)))


def get_optimizer(loss, opt_type="Adam", opt_params=None, optimizer=None, variables=None):
    if optimizer is None:
        learning_rate = opt_params.get("learning_rate", 1e-3)
        beta1 = opt_params.get("beta1", 0.9)
        beta2 = opt_params.get("beta2", 0.99)
        epsilon = opt_params.get("epsilon", 1e-8)
        if opt_type == "RMSProp":
            epsilon = opt_params.get("epsilon", 1e-10)
        rho = opt_params.get("rho", 0.95)
        global_step = opt_params.get("global_step", None)
        initial_gradient_squared_accumulator_value = opt_params.get("initial_gradient_squared_accumulator_value", 0.1)
        l1_regularization_strength = opt_params.get("l1_regularization_strength", 0.0)
        l2_regularization_strength = opt_params.get("l2_regularization_strength", 0.0)
        initial_accumulator_value = opt_params.get("initial_accumulator_value", 0.1)
        momentum = opt_params.get("momentum", 0.0)
        use_nesterov = opt_params.get("use_nesterov", False)
        decay = opt_params.get("decay", 0.9)
        if opt_type == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name=opt_type)
        elif opt_type == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho, epsilon=epsilon, name=opt_type)
        elif opt_type == "AdagradDA":
            if global_step is None:
                raise ValueError("global_step should be given in opt_params")
            optimizer = tf.train.AdagradDAOptimizer(learning_rate, global_step, initial_gradient_squared_accumulator_value=initial_gradient_squared_accumulator_value, l1_regularization_strength=l1_regularization_strength, l2_regularization_strength=l2_regularization_strength, name=opt_type)
        elif opt_type == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value, name=opt_type)
        elif opt_type == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate, name=opt_type)
        elif opt_type == "Momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, name=opt_type, use_nesterov=use_nesterov)
        elif opt_type == "ProximalAdagrad":
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate, initial_accumulator_value=initial_accumulator_value, l1_regularization_strength=l1_regularization_strength, l2_regularization_strength=l2_regularization_strength, name=opt_type)
        elif opt_type == "ProximalGradientDescent":
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate, l1_regularization_strength=l1_regularization_strength, l2_regularization_strength=l2_regularization_strength, name=opt_type)
        elif opt_type == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, epsilon=epsilon, name=opt_type)
    opt = optimizer.minimize(loss, global_step=global_step, var_list=variables)
    return opt


# # Mnist
# def GANloader(data_file, batch_size=128):
#     data = input_data.read_data_sets(data_file, one_hot=True)
#     while True:
#         batch = data.train.next_batch(batch_size)[0]
#         inputs = ((batch[0] - 0.5)*2)
#         inputs = inputs.reshape([-1, 28, 28, 1])
#         yield inputs
#
#
# def GANClassloader(data_file, batch_size=128):
#     data = input_data.read_data_sets(data_file, one_hot=True)
#     while True:
#         batch = data.train.next_batch(batch_size)
#         inputs = ((batch[0] - 0.5)*2)
#         inputs = inputs.reshape([-1, 28, 28, 1])
#         yield inputs, batch[1]


# image folder
def GANloader(folder, batch_size=128):
    filelist = [os.path.join(folder, file) for file in os.listdir(folder) if not os.path.isdir(os.path.join(folder, file))]
    num = len(filelist)
    while True:
        inputs = []
        for _ in range(batch_size):
            idx = np.random.randint(num)
            sample = mpimg.imread(filelist[idx])
            sample = (sample / 256 - 0.5) * 2
            inputs.append(sample)
        yield np.array(inputs)


def GANClassloader(folder, batch_size=128):
    folderlist = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isdir(os.path.join(folder, file))]
    filelists = []
    for fd in folderlist:
        filelists.append([os.path.join(fd, file) for file in os.listdir(fd) if os.path.isdir(os.path.join(fd, file))])
    num = len(folderlist)
    while True:
        inputs, labels = [], []
        for _ in range(batch_size):
            category = np.random.randint(num)
            idx = np.random.randint(len(filelists[category]))
            sample = mpimg.imread(filelists[category][idx])
            sample = (sample / 256 - 0.5) * 2
            inputs.append(sample)
            labels.append(category)
        yield np.array(inputs), np.eye(num)[labels]
