
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

def prep_tfrecord( image_file_label, tfrecord_destfile ):

    random.shuffle( image_file_label )

    list_length = len( image_file_label )
    print '{0} files to prepare'.format( list_length )

    writer = tf.python_io.TFRecordWriter(tfrecord_destfile)

    image_file_label_copy = image_file_label[:]
    random.shuffle( image_file_label_copy )

    for elem_one in image_file_label:

        file_name_one = elem_one[0]
        label_one     = elem_one[1]

        for elem_two in image_file_label_copy:
            file_name_two = elem_two[0]
            label_two     = elem_two[1]

            if file_name_one == file_name_two:
                continue

            if label_one == label_two:
                label = 0
            else:
                label = 1

        example = tf.train.Example(features=\
            tf.train.Features(feature={\
                'image_one': _bytes_feature(file_name_one),
                'image_two' :_bytes_feature( file_name_two ),
                'label': _int64_feature( [label] ),
            } ) )

        writer.write(example.SerializeToString())

    writer.close()

    print 'TFRecord file save: {0}'.format( tfrecord_destfile )

