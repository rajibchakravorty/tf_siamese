import cPickle
import random
from os.path import join,basename

from prep_tf_records import prep_tfrecord_class

def load_pickled_list( filename ):

    with open( filename, 'rb' ) as f:
        pickled_list = cPickle.load( f )

    return pickled_list

def count_class( image_label_list ):

    counts = [0,0,0,0,0,0,0,0,0,0]

    for il in image_label_list:

        counts[ il[1] ] = counts[ il[1] ] + 1

    return counts

def save_class_txt( class_list, outfile ):

    with open( outfile, 'wt' ) as f:

        for s in class_list:
            f.write( '{0},{1}\n'.format(s[0], s[1] ) )


if __name__ == '__main__':

    pickle_input = join( 'output','test.pickle' )
    output_file = join( 'output', 'test.txt' )
    tfrecord_file = join( 'output','test.tfrecords' )

    with open( pickle_input, 'rb' ) as f:
        class_valid_list = cPickle.load( f )

    save_class_txt( class_valid_list, output_file )


    prep_tfrecord_class( class_valid_list, tfrecord_file )





'''
    class_num = 10

    print 'Training/Valid 1/Valid 2/Test {0}/{1}/{2}/{3}'.format( len( train_list ),\
                                                                 len( same_valid ),\
                                                                 len( class_valid ),\
                                                                  len( test_list ))

    count_class(same_valid)
    count_class(class_valid)
    count_class(test_list)

    ##saving tf_records

    same_valid_file = 'same_valid.tfrecords'
    class_valid_file = 'class_valid.tfrecords'
    test_file = 'test.tfrecords'

    prep_tfrecord_siamese( same_valid, same_valid_file )
    prep_tfrecord_class( class_valid, class_valid_file )
    prep_tfrecord_class( test_list, test_file )


    for sample_per_class in [20,50,100,150,200,250,300,350,500]:
        total_samples = sample_per_class * class_num

        train_list_sampled = train_list[0:total_samples]
        count_class(train_list_sampled)
        train_file = 'train_{0}.tfrecords'.format(sample_per_class)
        prep_tfrecord_siamese(train_list_sampled, train_file)
'''