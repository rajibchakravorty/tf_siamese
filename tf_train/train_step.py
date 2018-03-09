'''
This file contains the typical steps of a classification task
'''

import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.contrib import layers

'''
function to prepare the validation summary op.
mean_loss and accuracy are tensors to be summarized
for later viewing in the tensorboard
'''
def prep_valid_summary_op( total_loss ):

    summary_collection = 'valid_summary'

    with tf.name_scope( summary_collection ) as scope:
        total_loss_op = tf.summary.scalar( 'mean_loss', total_loss,
                           collections=summary_collection )

    all_ops = [total_loss_op]
    summary_op = tf.summary.merge( all_ops, collections = summary_collection )

    return summary_op

'''
function to prepare the training summary op.
summarizes mean_loss, learning_rate, total_loss and accuracy
in addition, it stores the variable statistics (histogram
and distribution).
'''
def prep_train_summary_op( total_loss,
                           learning_rate):

    summary_collection = 'train_summary'

    with tf.name_scope( summary_collection ) as scope:


        total_loss_op = tf.summary.scalar( 'total_loss_summary', total_loss,
                               collections=summary_collection)

        learn_rate_op = tf.summary.scalar( 'learning_rate', learning_rate,
                           collections=summary_collection )


        # Add histograms for trainable variables.
        variable_ops = list()
        for var in tf.trainable_variables():
           variable_ops.append( tf.summary.histogram(var.op.name, var,
                                                      collections=summary_collection) )

    all_ops = [total_loss_op] + [learn_rate_op]+variable_ops
    summary_op = tf.summary.merge( all_ops, collections=summary_collection )

    return summary_op

'''
training steps of a typical classification task
Provision for supplying loss calculators and optimizers
'''
#TODO: test supplying other loss_op and optimizers

def contrastive_loss(output_1, output_2, label, margin):
    d = tf.sqrt(tf.reduce_sum(tf.pow(output_1 - output_2, 2), 1, keep_dims=True))
    ##original
    tmp= label * tf.square(d)
    tmp2 = (1 - label) * tf.square(tf.maximum((margin - d),0))
    #tmp= (1-label) * tf.square(d)
    #tmp2 = label * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(tmp + tmp2) /2

def train_step( images_one, images_two, labels, network,
                learning_rate_info, device_string,
                margin,
                optimizer,
                optimizer_kwargs,
                loss_collections=tf.GraphKeys.LOSSES,
                cpu_device = '/device:CPU:0'
                 ) :

    ##################################################################
    ##### training steps #############################################
    ##### fairly generic for most common classification tasks ########
    ##################################################################

    global_step_number = tf.train.create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay( learning_rate_info['init_rate'],
                                    global_step_number,
                                    learning_rate_info['decay_steps'],
                                    learning_rate_info['decay_factor'],
                                    learning_rate_info['staircase'])

    if optimizer_kwargs is None:
        updater = optimizer( learning_rate = learning_rate )
    else:
        updater = optimizer( learning_rate=learning_rate, **optimizer_kwargs )

    with tf.device( device_string ):
        ## get logits
        out_one = network( images_one, reuse = False )
        out_two = network( images_two, reuse = True )

    with tf.device( cpu_device ):

        labels_one_hot = tf.one_hot( labels,depth =2 )
        loss = contrastive_loss( out_one, out_two, labels_one_hot, margin )

        ##calculate gradient and apply it
        grads = updater.compute_gradients( loss )

        train_op = updater.apply_gradients( grads,
                                              global_step = global_step_number )
    ###########################################################################

    return out_one, out_two,loss, learning_rate, \
           global_step_number, train_op