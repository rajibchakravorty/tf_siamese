import operator
import numpy as np
from numpy import mean, std, dot, inner
from numpy.linalg import norm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import random

from os.path import join

import cPickle

sample_per_class = 1000

feature_location = join( '/home/rachakra/few_shot_learning/cifar-10',\
                         'sample_{0}'.format( sample_per_class ),\
                         'classification' )

test_feature_file = join( feature_location, 'test_features.pickle' )
train_feature_file = join( feature_location, 'train_feature.pickle' )


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
        train_feature = train_feature[0][0]
        train_feature = np.reshape( train_feature, (1,-1 ) )
        #print train_feature
        #print train_feature.shape
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
    train_files = _divide_by_labels(train_features)

    print len( test_features.keys() )
    print len( train_features.keys() )


    K = 10

    chosen_train_files = _choose_random( train_files, K )

    test_truth = []
    algo_label = []
    for idx, test_f in enumerate( test_features.keys()[0:200] ):

        print idx
        test_feature, test_label = test_features[test_f]
        test_feature = test_feature[0][0]
        test_feature = np.reshape( test_feature, (-1,1) )
        #print test_feature.shape

        chosen_distance = []
        for l in range( len( chosen_train_files ) ):

            train_files = chosen_train_files[l]

            all_distances, label = _get_cumul_distance_metric( test_feature, train_features, train_files )

            assert label == l, 'obtained label is not the same !!!'

            #print test_label, metric
            #min_index, max_index = _get_index( all_distances )

            chosen_distance.append( np.mean( all_distances ) )

        sorted_index = _get_sorted_index( chosen_distance )

            

        test_truth.append( test_label )
        if test_label in sorted_index[0:3]:
            algo_label.append( test_label)
        else:
            algo_label.append( sorted_index[0])



    
    #print test_truth
    print 'Accuracy: {0}'.format( accuracy_score( test_truth, algo_label ) )
    #print 'Min accuracy: {0}'.format( accuracy_score( test_truth, test_label_min ) )

      
