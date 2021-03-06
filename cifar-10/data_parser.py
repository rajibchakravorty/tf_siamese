import tensorflow as tf

default_contrast_lower = 0.3
default_contrast_upper = 0.8

class Parser():

    def __init__(self, features,
                 resize_height,
                 resize_width,
                 is_training = False ):
        self.features = features
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.is_training = is_training

    def parse_example( self, example_proto):

        parsed_features = tf.parse_single_example( example_proto, self.features )

        image_file_one = parsed_features['image_one']
        image_file_two = parsed_features['image_two']

        image_string = tf.read_file(image_file_one)
        image_one = tf.image.decode_jpeg(image_string,channels=3)
        image_one = tf.image.convert_image_dtype( image_one, tf.float32 )
        image_one = tf.image.per_image_standardization( image_one )

        image_string = tf.read_file(image_file_two)
        image_two = tf.image.decode_jpeg(image_string, channels=3)
        image_two = tf.image.convert_image_dtype(image_two, tf.float32)
        image_two = tf.image.per_image_standardization( image_two )

        #image = tf.image.rgb_to_grayscale( image )

        if self.is_training:

            image_one = tf.image.random_flip_up_down( \
                            tf.image.random_flip_left_right( image_one ) )

            image_two = tf.image.random_flip_left_right( \
                            tf.image.random_flip_up_down( image_two ) )



     

        label = tf.cast( parsed_features['label'], tf.int64 )


        #if for training, randomly translate the image
        # and then resize

        image_one = tf.image.resize_image_with_crop_or_pad( image_one, self.resize_height,
                                                        self.resize_width )
        image_two = tf.image.resize_image_with_crop_or_pad(image_two, self.resize_height,
                                                           self.resize_width)

        #return (image_file, image, height, width, channel, resize_height, resize_width)

        return image_one, image_two, label,image_file_one, image_file_two
