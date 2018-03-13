
import random

import numpy as np

import tensorflow as tf

import skimage
from skimage.io import imread

## expects a list
## each element is a tuple, (file_name, label)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def prep_tfrecord_class( image_file_label, tfrecord_destfile ):

    writer = tf.python_io.TFRecordWriter(tfrecord_destfile)
    for info in image_file_label:
        file_name = info[0]
        label = int(info[1])

        example = tf.train.Example(features= \
            tf.train.Features(feature={ \
                'image': _bytes_feature( file_name ),
                'label': _int64_feature([label]),
            }))

        writer.write(example.SerializeToString())

    writer.close()

    print 'TFRecord file save: {0}'.format(tfrecord_destfile)

def prep_tfrecord_siamese( image_file_label, tfrecord_destfile ):


    writer = tf.python_io.TFRecordWriter(tfrecord_destfile)

    for info in image_file_label:

        file_name_one = info[0]
        file_name_two = info[1]
        label  = int( info[2] )

        #print file_name_one, file_name_two

        example = tf.train.Example(features=\
            tf.train.Features(feature={\
                 'image_one': _bytes_feature(file_name_one),
                 'image_two' :_bytes_feature( file_name_two ),
                 'label': _int64_feature( [label] ),
        } ) )

        writer.write(example.SerializeToString())

    writer.close()

    print 'TFRecord file save: {0}'.format( tfrecord_destfile )

