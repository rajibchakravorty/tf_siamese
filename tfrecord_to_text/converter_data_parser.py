import tensorflow as tf

default_contrast_lower = 0.3
default_contrast_upper = 0.8

class SiameseParser():

    def __init__(self, features):
        self.features = features

    def parse_example( self, example_proto):

        parsed_features = tf.parse_single_example( example_proto, self.features )

        image_file_one = parsed_features['image_one']
        image_file_two = parsed_features['image_two']

        label = tf.cast( parsed_features['label'], tf.int64 )
        return image_file_one, image_file_two, label