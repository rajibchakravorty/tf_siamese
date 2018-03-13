from os.path import basename
import tensorflow as tf
from converter_data_preprocess import prepare_dataset
from converter_data_parser import SiameseParser

siamese_features={'image_one':tf.FixedLenFeature([], tf.string),\
    'image_two':tf.FixedLenFeature([], tf.string),\
    'label': tf.FixedLenFeature([], tf.int64 )}

train_parser = SiameseParser( siamese_features )
valid_parser = SiameseParser( siamese_features )

def get_image_label( image_file ):
    image_file_name_split = image_file.split('_')
    label_part = image_file_name_split[4]
    label_string = label_part.split('.')[0]

    return label_string

def prepare_lines_siamese( images_one, images_two, labels ):

    lines = []
    for im_one, im_two, lbl in zip( images_one, images_two, labels ):

        lines.append( '{0},{1},{2}'.format( im_one, im_two, lbl))

    return lines




def convert_siamese( input_tfrecord, outfile ):
    g = tf.Graph()

    with g.as_default():
        sess = tf.Session(graph=g )

        dataset = prepare_dataset(input_tfrecord,
                                           train_parser.parse_example,
                                           200)

        data_iterator = dataset.make_one_shot_iterator()
        next_element = data_iterator.get_next()

        ## from TF 1.4: this is a way to reuse one iterator
        ## for multiple datasets
        ###handle = tf.placeholder(tf.string, shape=[])
        ###data_iterator = tf.data.Iterator.from_string_handle(
        ###    handle, dataset.output_types, dataset.output_shapes)
        ###next_element = data_iterator.get_next()

        ##the train data iterator needs to be initialized one
        ## because it is infinite (.repeat()
        ## validation data iterator is initializable multiple times
        ###data_iterator = dataset.make_one_shot_iterator()

        sess.run(tf.global_variables_initializer())
        ##data_handle = sess.run(data_iterator.string_handle())


        while True:
            f = open( outfile, 'w' )
            try:
                image_one, image_two, label = sess.run(next_element)

                for im_one, im_two, lbl in zip(image_one, image_two, label):
                    print '{0},{1},{2}'.format( basename( im_one ), basename( im_two ), lbl)
            except tf.errors.OutOfRangeError:
                f.close()
                break
            #f.close()

if __name__ == '__main__':
    input_tfrecord = '/home/deeplearner/progs/few_shot/prepare_data/train_20.tfrecords'
    outfile_text = 'train_20.txt'
    convert_siamese( input_tfrecord, outfile_text)