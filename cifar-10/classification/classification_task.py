
from numpy import dot, inner
from numpy.linalg import norm

import random

from os.path import join

import cPickle

feature_location = '/Users/rachakara/progs/few_shots_experiments/few_shot/cifar-10/classification_50/'

test_feature_file = join( feature_location, 'test_features.pickle' )
train_feature_file = join( feature_location, 'train_feature.pickle' )

K = 3

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


    for test_f in test_features.keys()[1012:1013]:

        test_feature, test_label = test_features[test_f]
        test_feature = test_feature[0]

        all_distances = list()
        for l in range( len( chosen_train_files ) ):

            train_files = chosen_train_files[l]

            label_distances = list()
            for train_f in train_files:

                train_feature, train_label = train_features[train_f]
                train_feature = train_feature[0]
                assert train_label == l, 'obtained label is not the same'
                dist = inner( test_feature, train_feature ) / (norm( train_feature ) * norm(test_feature ) )

                label_distances.append( dist )

            print label_distances
            #print len( label_distances )
            all_distances.append( label_distances )



        #decide_class( all_distances )
