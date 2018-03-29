
from os.path import join
import cPickle
import random


random.seed( 2034 )

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

    #data_location = '/Users/rachakara/progs/few_shots_experiments/images/cifar-10-batches-py'
    data_location = '/home/rachakra/data/ml_data/cifar/cifar-10-batches-py'

    image_location = '/home/rachakra/data/ml_data/cifar/train'

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
    same_valid = valid_list[0:1000]
    class_valid = valid_list[5000:]

    print len(valid_list), len( train_list ), len( same_valid), len( class_valid)
    

    #image_location = '/Users/rachakara/progs/few_shots_experiments/images/test'
    image_location = '/home/rachakra/data/ml_data/cifar/test'
    batch_file = join(data_location, 'test_batch')
    test_list = get_batch_data_lists(batch_file, 'test', image_location)

    ## saving the lists

    train_list_file = join( 'output','train.pickle')
    siamese_valid_file = join( 'output','siamese_valid.pickle' )
    class_valid_file = join( 'output','class_valid.pickle' )
    test_file        = join( 'output','test.pickle' )

    with open( train_list_file, 'wb' ) as f:
        cPickle.dump(train_list, f )

    with open( siamese_valid_file, 'wb' ) as f:
        cPickle.dump(same_valid, f )

    with open(class_valid_file, 'wb') as f:
        cPickle.dump(class_valid, f)

    with open(test_file, 'wb') as f:
        cPickle.dump(test_list, f)

   
