
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

from os.path import join

import tensorflow as tf

from cnn_hsu import cnn_archi as network

from data_parser import Parser

sample_per_class = 1000

sample_path = join( '/home/rachakra/few_shot_learning/cifar-10',\
                    'sample_{0}'.format( sample_per_class ) )

checkpoint_path = join( sample_path, 'checkpoints_hsu' )
prior_weights = None #join( checkpoint_path, 'model.ckpt-00006000' )
classification_path = join( sample_path, 'classification' )

test_prior_weights = join( checkpoint_path, 'model.ckpt-00126000' )
#test_prior_weights = join( checkpoint_path, 'model.ckpt-00021000' )

device_string = '/gpu:0'

## definition of epoch in terms of batch number
batch_per_epoch = 3000 #40000-500
batch_size = 32

## batches to be used during statistics collections
batch_per_test = 3000 #3000


learning_rate_info = dict()
learning_rate_info['init_rate'] = 0.00005 #0.00005
learning_rate_info['decay_steps'] = 200 * batch_per_epoch
learning_rate_info['decay_factor'] = 0.95
learning_rate_info['staircase']  =True

##loss operations
loss_op=tf.losses.sparse_softmax_cross_entropy
one_hot=False
loss_op_kwargs = None
contrastive_margin = 2.0

##optimizers
optimizer = tf.train.RMSPropOptimizer
optimizer_kwargs = None

image_height = 32
image_width  = 32
image_channel  = 3

class_numbers = 10

#model storage

model_checkpoint_path = join( checkpoint_path, 'model.ckpt')
train_summary_path = join( checkpoint_path, 'train' )
valid_summary_path = join( checkpoint_path, 'valid' )


## data loading
root_path = '/home/rachakra/few_shot_learning/prepare_data/output'
train_tfrecords = join( root_path, 'train_siamese_pair_{0}.tfrecords'.format( sample_per_class ) )
valid_tfrecords = join( root_path, 'siamese_pair_valid.tfrecords' )

## information for parsing the tfrecord

features={'image_one':tf.FixedLenFeature([], tf.string),
    'image_two':tf.FixedLenFeature([], tf.string),\
    'label': tf.FixedLenFeature([], tf.int64 )}

train_parser = Parser( features, image_height, image_width, True )
valid_parser = Parser( features, image_height, image_width, False )


##test files
training_list_file = 'train_raw_list_{0}.txt'.format( sample_per_class )
class_valid_list = join( root_path, 'class_valid.pickle' )
