
from os.path import join
import cPickle
import random

from prep_tf_records import prep_tfrecord

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)

    images = dict['data']
    labels = dict['labels']

    return images, labels

def get_batch_data_lists( data_file, batch, image_path ):

    images, labels = unpickle( data_file )

    data_list = list()
    for idx, im in enumerate( images ):

        label = labels[idx]

        name = 'im_{0}_{1}_{2}.jpg'.format(batch, idx, label)

        image_file = join( image_path, name )

        data_list.append ((image_file, label ) )


    return data_list

def count_class( image_label_list ):

    counts = [0,0,0,0,0,0,0,0,0,0]

    for il in image_label_list:

        counts[ il[1] ] = counts[ il[1] ] + 1

    print counts

if __name__ == '__main__':

    data_location = '/opt/ml_data/cifar/cifar-10-batches-py'


    image_location = '/opt/ml_data/cifar/cifar-10/images/train'

    ##batch_1
    batch_file = join( data_location, 'data_batch_1' )
    batch_1_list = get_batch_data_lists(batch_file, 1, image_location)

    batch_file = join(data_location, 'data_batch_2')
    batch_2_list = get_batch_data_lists(batch_file, 2, image_location)

    batch_file = join(data_location, 'data_batch_3')
    batch_3_list = get_batch_data_lists(batch_file, 3, image_location)

    batch_file = join(data_location, 'data_batch_4')
    batch_4_list = get_batch_data_lists(batch_file, 4, image_location)

    batch_file = join(data_location, 'data_batch_5')
    batch_5_list = get_batch_data_lists(batch_file, 5, image_location)

    train_list = batch_1_list+batch_2_list+batch_3_list+batch_4_list+batch_5_list

    random.shuffle(train_list)

    valid_list = train_list[-10000:]
    train_list = train_list[0:-10000]
    same_valid = valid_list[0:5000]
    class_valid = valid_list[5000:]

    image_location = '/opt/ml_data/cifar/cifar-10/images/test'
    batch_file = join(data_location, 'test_batch')
    test_list = get_batch_data_lists(batch_file, 'test', image_location)

    class_num = 10

    print 'Training/Valid 1/Valid 2/Test Samples {0}/{1}/{2}/{3}'.format( len( train_list ),\
                                                                 len( same_valid ),\
                                                                 len( class_valid ),\
                                                                 len( test_list ) )

    print test_list[0][0]
    count_class(same_valid)
    count_class(class_valid)
    count_class(test_list)

    ##saving tf_records

    same_valid_file = 'same_valid.tfrecords'
    class_valid_file = 'class_valid.tfrecords'
    test_valid_file = 'test.tfrecords'

    prep_tfrecord( same_valid, same_valid_file )
    prep_tfrecord( class_valid, class_valid_file )
    prep_tfrecord( test_list, test_valid_file )


    for sample_per_class in [20,50,100,150,200,250,300,350,500]:
        total_samples = sample_per_class * class_num

        train_list_sampled = train_list[0:total_samples]
        count_class(train_list_sampled)
        train_file = 'train_{0}.tfrecords'.format(sample_per_class)
        prep_tfrecord(train_list_sampled, train_file)
