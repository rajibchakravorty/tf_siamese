'''
This file contains the typical steps of a classification task
'''

import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.contrib import layers

def test_step( images_one, network) :


    tf.expand_dims( images_one, 0 )

    output = network( images_one )
        #output = network( images_one, images_two )


    return output
    #return output, loss, learning_rate, global_step_number, train_op