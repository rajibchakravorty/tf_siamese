

import tensorflow as tf


def cnn_archi(input_one, reuse=tf.AUTO_REUSE):

    
    with tf.variable_scope("conv1") as scope:
        net = tf.contrib.layers.conv2d(input_one, 10, [3,3],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       #weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5),
                                       scope=scope, 
                                       reuse=reuse)

        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, 0.5 )
    #net = tf.contrib.layers.batch_norm( net)

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net, 10, [5, 5],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       #weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5),
                                       scope=scope, reuse=reuse)

        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    #net = tf.contrib.layers.dropout( net, 0.5 )
    #net = tf.contrib.layers.batch_norm( net )
    '''
    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 128, [3, 3],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       #weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5),
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, 0.5 )
    #net = tf.contrib.layers.batch_norm( net )
   
    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net, 256, [1, 1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       #weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5),
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, 0.5 )
    #net = tf.contrib.layers.batch_norm( net, reuse )

    with tf.variable_scope("conv5") as scope:
        net = tf.contrib.layers.conv2d(net, 1024, [1, 1],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       #weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.5, scale_l2=0.5),
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, 0.5 )
    #net = tf.contrib.layers.batch_norm( net, reuse )
    '''
    net = tf.contrib.layers.flatten(net)

    #with tf.variable_scope( "fc1" ) as scope:

    #    net = tf.layers.dense ( tf.layers.dropout( net, 0.5) , 1024, activation = tf.nn.relu,\
    #                            reuse=reuse)
    #with tf.variable_scope("fc2") as scope:
    #    net = tf.layers.dense( tf.layers.dropout( net,0.5) , 128, activation=tf.nn.relu, \
    #                          reuse=reuse)

    #with tf.variable_scope("fc3") as scope:
    #    net = tf.layers.dense( tf.layers.dropout( net,0.5) , 2, activation=tf.nn.relu, \
    #                          reuse=reuse)

    #with tf.variable_scope("fc4") as scope:
    #    net = tf.layers.dense( net , 2, activation=tf.nn.softmax, \
    #                          reuse=reuse)


    return net
