

import tensorflow as tf


def cnn_archi(input_one, dropout_prob, reuse=tf.AUTO_REUSE):

    #net = tf.contrib.layers.batch_norm( input_one )
    with tf.variable_scope("conv1") as scope:
        net = tf.contrib.layers.conv2d(input_one, 96, [5,5],
                                       #stride = [1,1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
                                       scope=scope, 
                                       reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net, [3, 3], padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    #net = tf.contrib.layers.batch_norm( net)

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net,192, [3, 3],
                                       #stride = [1,1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
                                       scope=scope, reuse=reuse)

        net = tf.contrib.layers.max_pool2d(net,[2,2],padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    #net = tf.contrib.layers.batch_norm( net )
    
    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 384, [3, 3],#stride=[1,1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    #net = tf.contrib.layers.batch_norm( net )
    
    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net,128, [2, 2],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1 =0.02, scale_l2=0.02 ),
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [3,3],stride=[2,2],padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    #net = tf.contrib.layers.batch_norm( net )
   
      
    with tf.variable_scope("conv5") as scope:
        net = tf.contrib.layers.conv2d(net, 64, [1, 1],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    net = tf.contrib.layers.batch_norm( net )

    #with tf.variable_scope("conv6") as scope:
    #    net = tf.contrib.layers.conv2d(net,20, [3, 3],
    #                                   activation_fn=None, padding='SAME',
    #                                   weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                   weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
    #                                   scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        #net = tf.contrib.layers.dropout( net, keep_prob = dropout_prob, scope=scope )
    #net = tf.contrib.layers.batch_norm( net )
       
    net = tf.contrib.layers.flatten(net)
    

    return net
