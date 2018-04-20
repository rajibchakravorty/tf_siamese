

import tensorflow as tf


def cnn_archi(input_one, dropout_prob, reuse=tf.AUTO_REUSE):

    net = tf.contrib.layers.batch_norm( input_one )
    with tf.variable_scope("conv1") as scope:
        net = tf.contrib.layers.conv2d(input_one, 10, [3,3],
                                       stride = [2,2],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.0002, scale_l2=0.0002 ),\
                                       scope=scope, 
                                       reuse=reuse)

        #net = tf.contrib.layers.max_pool2d(net, [3, 3],stride=[2,2], padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    #net = tf.contrib.layers.batch_norm( net)

    with tf.variable_scope("conv2") as scope:
        net = tf.contrib.layers.conv2d(net, 10, [3, 3],
                                       stride = [1,1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.0002, scale_l2=0.0002 ),\
                                       scope=scope, reuse=reuse)

        #net = tf.contrib.layers.max_pool2d(net,[3,3],stride= [2, 2],padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    #net = tf.contrib.layers.batch_norm( net )
    
    with tf.variable_scope("conv3") as scope:
        net = tf.contrib.layers.conv2d(net, 10, [3, 3],stride=[1,1],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.0002, scale_l2=0.0002 ),\
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    #net = tf.contrib.layers.batch_norm( net )
    
    with tf.variable_scope("conv4") as scope:
        net = tf.contrib.layers.conv2d(net,10, [3, 3],
                                       activation_fn=tf.nn.relu, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer=tf.contrib.layers.l2_regularizer(scale =0.002 ),
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2],padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    #net = tf.contrib.layers.batch_norm( net )
   
    ''' 
    with tf.variable_scope("conv5") as scope:
        net = tf.contrib.layers.conv2d(net, 10, [3, 3],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.0002, scale_l2=0.0002 ),\
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    #net = tf.contrib.layers.batch_norm( net, reuse )

    with tf.variable_scope("conv6") as scope:
        net = tf.contrib.layers.conv2d(net,10, [3, 3],
                                       activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.0002, scale_l2=0.0002 ),\
                                       scope=scope, reuse=reuse)
        #net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
    net = tf.contrib.layers.dropout( net, dropout_prob )
    '''
    net = tf.contrib.layers.flatten(net)
    

    with tf.variable_scope( "fc1" ) as scope:

        net = tf.contrib.layers.fully_connected( net , 1024, activation_fn = tf.nn.relu,\
                                weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.002, scale_l2=0.002 ),\
                                reuse=reuse, scope = scope)
        #net  = tf.layers.dropout( net, dropout_prob )
    #with tf.variable_scope("fc2") as scope:
    #    net = tf.contrib.layers.fully_connected( net , 192, activation_fn = tf.nn.relu,\
    #                            reuse=reuse, scope = scope )
    #with tf.variable_scope("fc3") as scope:
    #    net = tf.contrib.layers.fully_connected( tf.layers.dropout( net, 0.5) , 64, activation_fn = tf.nn.leaky_relu,\
    #                            weights_regularizer = tf.contrib.layers.l1_l2_regularizer( scale_l1=0.02, scale_l2=0.02 ),\
    #                            reuse=reuse, scope = scope )

    '''
    with tf.variable_scope("fc4") as scope:
        net = tf.layers.dense( net , 2, activation=tf.nn.softmax, \
                              reuse=reuse)
    '''

    return net
