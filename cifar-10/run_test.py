from os.path import join

import numpy as np
from skimage.io import imread
from skimage import img_as_float

import cPickle
from tf_test.tester import Tester

#import sample_1000.config_lewis as config
import sample_1000.config_hoffer as config



def _read_file_label( list_pickle_file):

    return cPickle.load( open( list_pickle_file) )

def _read_image( image_file ):

    image_array = img_as_float( imread( image_file ) )
    return image_array


def _get_file_feature( image_file, tester ):

    return tester.run_test( image_file )

def _get_files_feature( file_list, tester ):

    result = dict()

    for f in file_list:

        image_file = f[0]
        label      = f[1]

        feature = _get_file_feature( image_file, tester )
        print feature[0].shape
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

    root_data_folder = '/home/rachakra/few_shot_learning/prepare_data/output'
    train_image_folder = '/home/rachakra/data/ml_data/cifar/train/'

    train_file = join( root_data_folder, config.training_list_file )

    file_label_list = _read_file_label( config.class_valid_list )


    tester = Tester( config.network, config )

    train_list = _text_to_list( train_file, train_image_folder)

    class_test_result = _get_files_feature( file_label_list, tester )
    train_result = _get_files_feature( train_list, tester)

    print len( class_test_result.keys() )
    print len( train_result.keys() )
    
    output_folder = config.classification_path

    test_features_file = join( output_folder, 'test_features.pickle' )
    train_features_file = join( output_folder, 'train_feature.pickle' )
    
    with open( test_features_file,'wb' ) as f:

        cPickle.dump( class_test_result, f )

    with open(train_features_file, 'wb') as f:
        cPickle.dump( train_result, f)
   
