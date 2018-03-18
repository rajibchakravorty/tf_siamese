
import numpy as np
from skimage.io import imread
from skimage import img_as_float

import cPickle

import config as config

from tf_test.tester import Tester
from cnn_2 import cnn_archi as network


def _read_file_label( list_pickle_file):

    return cPickle.load( open( list_pickle_file) )

file_label_list = _read_file_label( config.class_valid_list )

file = file_label_list[4800][0]


image_array = img_as_float( imread(  file ) )

tester = Tester(network, config )

print tester.run_test( image_array )