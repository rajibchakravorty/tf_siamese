
import numpy as np
from numpy import mean, std, dot, inner
from numpy.linalg import norm

from sklearn.metrics.pairwise import cosine_similarity

import random

from os.path import join

import cPickle

feature_location = '/home/rachakra/few_shot_learning/cifar-10/classification/'

test_feature_file = join( feature_location, 'test_features.pickle' )
train_feature_file = join( feature_location, 'train_feature.pickle' )

K = 10

random.seed( 2440)

def _read_features( feature_file ):

    return cPickle.load( open(feature_file, 'rb' ) )

def _divide_by_labels( feature_list,labels = 10,  ):


    label_features = [list() for _ in range( labels )]
    for f in feature_list.keys():

        label = feature_list[f][1]

        label_features[label].append( f )

    return label_features

def _choose_random( train_files, chosen_K ):

    labels = len( train_files )
    chosen_files = [list() for _ in range(labels)]
    for idx, f in enumerate( train_files ):

        random.shuffle( f )
        if chosen_K > len( f ):
            chosen_K = len(f)
        chosen_files[idx] = f[0:chosen_K]

    return chosen_files


def _get_distance( test_feature, train_features ):

    all_distances = list()
    for train_f in train_features:
        dist = cosine_similarity( test_feature, train_f )
        all_distances.append( dist )

    all_distances = np.array( all_distances )
    return all_distances
   
def _get_metric( all_distances ):

    mean_dist = mean( all_distances )
    var_std = std( all_distances )

    t = mean_dist / var_std

    return t

def _get_cumul_distance_metric( test_feature, train_features, train_files ):

    all_features = list()
    for train_f in train_files:

        train_feature, train_label = train_features[train_f]
        train_feature = train_feature[0]
        #print train_feature
        all_features.append( train_feature )

    all_distances = _get_distance( test_feature, all_features)
    metric = _get_metric( all_distances )

    return metric, train_label

if __name__ == '__main__':

    test_features = _read_features( test_feature_file )
    train_features = _read_features( train_feature_file )

    test_files = _divide_by_labels(test_features)
    '''
    print len(test_files)

    for l in test_files:
        print 'Class {0}'.format( len( l ) )
        print l[0]
    '''
    train_files = _divide_by_labels(train_features)

    '''
    print len(train_files)

    for l in train_files:
        print 'Class {0}'.format( len( l ) )
    '''

    chosen_train_files = _choose_random( train_files, K )
    '''
    print len( chosen_train_files )

    for c in chosen_train_files:

        print len( c )
        print c[0]

    '''


    for test_f in test_features.keys()[0:5]:

        test_feature, test_label = test_features[test_f]
        test_feature = test_feature[0]

        print test_label

     
        for l in range( len( chosen_train_files ) ):

            train_files = chosen_train_files[l]

            metric, label = _get_cumul_distance_metric( test_feature, train_features, train_files )

            assert label == l, 'obtained label is not the same !!!'

            print test_label, metric



        #decide_class( all_distances )
       
