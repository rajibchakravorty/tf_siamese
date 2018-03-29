
from os.path import join

import tensorflow as tf


from data_parser import Parser

device_string = '/device:GPU:0'

## definition of epoch in terms of batch number
batch_per_epoch = 2000 #3000 #40000-500
batch_size = 32

## batches to be used during statistics collections
batch_per_test = 300 #3000


learning_rate_info = dict()
learning_rate_info['init_rate'] = 0.0005 #0.00005
learning_rate_info['decay_steps'] = 60*2000
learning_rate_info['decay_factor'] = 0.95
learning_rate_info['staircase']  =True

##loss operations
loss_op=tf.losses.sparse_softmax_cross_entropy
one_hot=False
loss_op_kwargs = None
contrastive_margin = 0.5

##optimizers
optimizer = tf.train.AdamOptimizer
optimizer_kwargs = None

image_height = 32
image_width  = 32
image_channel  = 3

class_numbers = 10

checkpoint_path = './checkpoints'
model_checkpoint_path = join( checkpoint_path, 'model.ckpt')
prior_weights = None #join( checkpoint_path, 'model.ckpt-00060000' )
train_summary_path = join( checkpoint_path, 'train' )
valid_summary_path = join( checkpoint_path, 'valid' )



#root_path = '/home/deeplearner/progs/few_shot/prepare_data/output'
#root_path = '/Users/rachakara/progs/few_shots_experiments/few_shot/prepare_data/output'
root_path = '/home/rachakra/few_shot_learning/prepare_data/output'
sample_per_class = 500 ##20,50,100,200,300
train_tfrecords = join( root_path, 'train_siamese_pair_{0}.tfrecords'.format( sample_per_class ) )
valid_tfrecords = join( root_path, 'siamese_pair_valid.tfrecords' )

## information for parsing the tfrecord
features={'image_one':tf.FixedLenFeature([], tf.string),
    'image_two':tf.FixedLenFeature([], tf.string),\
    'label': tf.FixedLenFeature([], tf.int64 )}



train_parser = Parser( features, image_height, image_width )
valid_parser = Parser( features, image_height, image_width )


##test files
test_checkpoint_path = './checkpoints'
test_prior_weights = join( checkpoint_path, 'model.ckpt-00140000' )
class_valid_list = join( root_path, 'class_valid.pickle' )
