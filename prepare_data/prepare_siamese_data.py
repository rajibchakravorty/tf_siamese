import cPickle
import random
from os.path import join,basename

from prep_tf_records import  prep_tfrecord_siamese

random.seed( 2035 )

def load_pickled_list( filename ):

    with open( filename, 'rb' ) as f:
        pickled_list = cPickle.load( f )

    return pickled_list

def count_class( image_label_list ):

    counts = [0,0,0,0,0,0,0,0,0,0]

    for il in image_label_list:

        counts[ il[1] ] = counts[ il[1] ] + 1

    return counts

def random_combination( image_label_list ):

    same_file_list = []
    diff_file_list = []

    image_label_copy = image_label_list[:]
    random.shuffle( image_label_copy )
    idx = 0
    for info_one in image_label_list[idx:]:

        for info_two in image_label_copy[(idx+1):]:

            if info_one[0] == info_two[0]:
                continue

            if info_one[1] == info_two[1]:

                same_file_list.append( ( info_one[0], info_two[0], 0 ) )

        idx += 1

    max_same_example = len( same_file_list )
    print 'Maximum same example : {0}'.format(max_same_example)

    random.shuffle(image_label_copy)
    idx = 0
    for info_one in image_label_list:

        for info_two in image_label_copy:

            if info_one[0] == info_two[0]:
                continue

            if info_one[1] != info_two[1]:
                diff_file_list.append((info_one[0], info_two[0], 1))

                if len( diff_file_list ) >= max_same_example:
                    break

        if len( diff_file_list ) >= max_same_example:
            break

    all_samples = same_file_list + diff_file_list
    random.shuffle( all_samples )

    return all_samples

def choose_random_list( info_list, N, outfile ):

    random.shuffle( info_list )

    selected_list = info_list[0:N]

    with open( outfile, 'w' ) as f:
        for l in selected_list:
            f.write( '{0},{1}\n'.format( basename( l[0] ), l[1] ) )

    class_count  = count_class( selected_list )

    return class_count, selected_list

def save_siamese_txt( siamese_list, outfile ):

    with open( outfile, 'wt' ) as f:

        for s in siamese_list:
            f.write( '{0},{1},{2}\n'.format(s[0], s[1], s[2] ) )


if __name__ == '__main__':

    
    per_class_samples = [20,50,100,200,300,500]

    for per_class_sample in per_class_samples:

        pickle_input = join( 'output','train.pickle' )
        output_file = join( 'output', 'train_raw_list_{0}.txt'.format( per_class_sample ) )
        output_siamese_file = join( 'output','train_siamese_pair_{0}.txt'.format( per_class_sample ) )
        tfrecord_file = join( 'output','train_siamese_pair_{0}.tfrecords'.format( per_class_sample ) )

        class_count, selected_list = choose_random_list( load_pickled_list( pickle_input ),
                                                         10*per_class_sample, output_file)

        print class_count
        print selected_list[0:1]

        selected_siamese_list = random_combination( selected_list )

        print selected_siamese_list[0]
        save_siamese_txt( selected_siamese_list, output_siamese_file )

        prep_tfrecord_siamese( selected_siamese_list, tfrecord_file )

   

    pickle_input = join('output', 'siamese_valid.pickle')
    output_file = join('output', 'siamese_valid_raw_list.txt' )
    output_siamese_file = join('output', 'siamese_pair_valid.txt' )
    tfrecord_file = join('output', 'siamese_pair_valid.tfrecords' )

    class_count, selected_list = choose_random_list(load_pickled_list(pickle_input),
                                                    1000, output_file)

    print class_count
    print selected_list[0:1]

    selected_siamese_list = random_combination(selected_list)

    print selected_siamese_list[0]
    save_siamese_txt(selected_siamese_list, output_siamese_file)

    prep_tfrecord_siamese(selected_siamese_list, tfrecord_file)

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
