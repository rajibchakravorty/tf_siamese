

import tensorflow as tf


def cnn_archi(input, reuse=False):

    with tf.variable_scope("conv1") as scope:
        net = tf.contrib.layers.conv2d(input, 32, [7, 7],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net, 64, [5, 5],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 128, [3, 3],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net, 256, [1, 1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv5") as scope:
        net = tf.contrib.layers.conv2d(net, 2, [1, 1],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    net = tf.contrib.layers.flatten(net)

    return net

