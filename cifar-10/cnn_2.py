

import tensorflow as tf


def cnn_archi(input_one, reuse=tf.AUTO_REUSE):

    with tf.variable_scope("conv1") as scope:
        net = tf.contrib.layers.conv2d(input_one, 8, [7, 7],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net, 16, [5, 5],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 32, [3, 3],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net, 64, [1, 1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    with tf.variable_scope("conv5") as scope:
        net = tf.contrib.layers.conv2d(net, 128, [1, 1],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       scope=scope, reuse=reuse)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

    net = tf.contrib.layers.flatten(net)

    with tf.variable_scope( "output" ) as scope:

        net = tf.layers.dense ( tf.layers.dropout( net, 0.5) , 1024, activation = tf.nn.relu,\
                                reuse=reuse)
    with tf.variable_scope("softmax_out") as scope:
        net = tf.layers.dense( tf.layers.dropout( net,0.5) , 10, activation=tf.nn.relu, \
                              reuse=reuse)


    return net

