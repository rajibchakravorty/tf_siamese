from os.path import join

import numpy as np
from skimage.io import imread
from skimage import img_as_float

import cPickle

import config as config

from tf_test.tester import Tester
from cnn_2 import cnn_archi as network


def _read_file_label( list_pickle_file):

    return cPickle.load( open( list_pickle_file) )

def _read_image( image_file ):

    image_array = img_as_float( imread( image_file ) )
    return image_array

def _get_image_feature( im, tester ):

    return tester.run_test( im )

def _get_file_feature( image_file, tester ):

    return _get_image_feature( _read_image( image_file), tester )

def _get_files_feature( file_list, tester ):

    result = dict()

    for f in file_list:

        image_file = f[0]
        label      = f[1]

        feature = _get_file_feature( image_file, tester )
        result[image_file] = (feature, label)

    return result

def _text_to_list( text_file_name ,root_data_folder ):

    result_list = list()
    with open( text_file_name, 'rb' ) as f:

        lines = f.readlines()

        for l in lines:

            file_name, label = l.split( ',' )
            result_list.append( ( join(root_data_folder, file_name),\
                                  int( label ))  )

    return result_list

if __name__ == '__main__':

    root_data_folder = '/Users/rachakara/progs/few_shots_experiments/few_shot/prepare_data/output'
    root_image_folder = '/Users/rachakara/progs/few_shots_experiments/images/train'

    train_file = join( root_data_folder, 'train_raw_list_50.txt' )

    file_label_list = _read_file_label( config.class_valid_list )


    tester = Tester(network, config )

    train_list = _text_to_list( train_file, root_image_folder)

    class_test_result = _get_files_feature( file_label_list, tester )
    train_result = _get_files_feature( train_list, tester)

    print len( class_test_result.keys() )
    print len( train_result.keys() )

    output_folder = '/Users/rachakara/progs/few_shots_experiments/few_shot/cifar-10/classification_50/'

    test_features_file = join( output_folder, 'test_features.pickle' )
    train_features_file = join( output_folder, 'train_feature.pickle' )

    with open( test_features_file,'wb' ) as f:

        cPickle.dump( class_test_result, f )

    with open(train_features_file, 'wb') as f:
        cPickle.dump( train_result, f)

