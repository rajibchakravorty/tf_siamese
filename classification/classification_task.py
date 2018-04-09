import operator
import numpy as np
from numpy import mean, std, dot, inner
from numpy.linalg import norm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import random

from os.path import join

import cPickle

feature_location = '/home/rachakra/few_shot_learning/cifar-10/classification/'

test_feature_file = join( feature_location, 'test_features.pickle' )
train_feature_file = join( feature_location, 'train_feature.pickle' )

K = 3

#random.seed( 2440)

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
        dist = np.sum( np.power( test_feature-train_f, 2.0 ) ) #cosine_similarity( test_feature, train_f )
        #dist = np.abs( cosine_similarity( test_feature, train_f ) )
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
  
    #print np.max( all_distances) , np.min( all_distances ) 

    return all_distances, train_label

def _get_index( input_list ):

    
    min_index, _ = min( enumerate( input_list ), key = operator.itemgetter( 1 ) )
    max_index, _ = max( enumerate( input_list ), key = operator.itemgetter( 1 ) )
    
    return min_index, max_index

def _get_sorted_index( input_list ):

    input_list_copy = input_list[:]
    return sorted( range( len(input_list_copy)), key=lambda k:input_list_copy[k])
    


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

    test_truth = []
    test_label_max = []
    test_label_min = []
    for test_f in test_features.keys():

        test_feature, test_label = test_features[test_f]
        test_feature = test_feature[0]

        print test_label

        
        for l in range( len( chosen_train_files ) ):

            train_files = chosen_train_files[l]

            all_distances, label = _get_cumul_distance_metric( test_feature, train_features, train_files )

            assert label == l, 'obtained label is not the same !!!'

            #print test_label, metric
            #min_index, max_index = _get_index( all_distances )
            sorted_index = _get_sorted_index( all_distances )

            

            test_truth.append( test_label )
            #test_label_max.append( max_index )
            #test_label_min.append( min_index )
            if test_label in sorted_index[0:3]:
                test_label_max.append( test_label)
            else:
                test_label_max.append( sorted_index[0])



    
    #print test_truth
    print 'Max accuracy: {0}'.format( accuracy_score( test_truth, test_label_max ) )
    #print 'Min accuracy: {0}'.format( accuracy_score( test_truth, test_label_min ) )

       
