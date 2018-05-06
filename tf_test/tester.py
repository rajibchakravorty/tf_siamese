'''
implements the training and validation loops
'''
import numpy as np


import tensorflow as tf

#from tf_test.test_step import test_step


class Tester():

    def __init__(self, network, config):

        ## network definition comes from client
        ## makes this part independent
        self.network     = network
        self.config      = config
        #self.file_list   = self.config.file_list
        self.model_file  = self.config.test_prior_weights

        print self.model_file
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True

        self.gpu_config = gpu_config
        self.output, self.session, self.image_file, self.dropout, self.image = self._prepare_graph_session()
        print 'Model loaded...'


    def _prepare_graph_session2(self):

        tf.reset_default_graph()

        sess = tf.Session()
   
        #meta_file = '{0}.meta'.format( self.model_file )
        saver = tf.train.import_meta_graph( '{0}.meta'.format( self.model_file ) )
        sess.run(tf.global_variables_initializer())
        saver = saver.restore( sess, self.model_file )
       

        g = tf.get_default_graph()

        with g.as_default():
            image_file = tf.placeholder( tf.string )


            image_string = tf.read_file(image_file)
            
            image = tf.image.decode_jpeg(image_string,channels=3)
            image = tf.image.convert_image_dtype( image, tf.float32 )
            image = tf.image.per_image_standardization( image )


            #image_placeholder= tf.placeholder( tf.float32,
            #                                   shape = (self.config.image_height,
            #                                            self.config.image_width,
            #                                            self.config.image_channel ) )

            image.set_shape( [self.config.image_height,
                              self.config.image_width,
                              self.config.image_channel] )
            image_input = tf.expand_dims( image, axis = 0)
            #output = test_step( image_input, self.network )

            with tf.device( self.config.device_string ) as scope:          
                output = self.network( image_input, dropout_prob )

            #output_norm = tf.divide( output, tf.norm( output, ord=2, axis = 1, keep_dims = True ) )
            #sess.run(tf.global_variables_initializer())


        return output, sess, image_file, image_input


    def _prepare_graph_session(self):

        tf.reset_default_graph()

        

        g = tf.Graph()

        with g.as_default():

            sess = tf.Session(graph=g,config = self.gpu_config)

            dropout_placeholder = tf.placeholder( tf.float32, shape =  () )
            image_file = tf.placeholder( tf.string )


            image_string = tf.read_file(image_file)
            
            image = tf.image.decode_jpeg(image_string,channels=3)
            image = tf.image.convert_image_dtype( image, tf.float32 )
            image = tf.image.per_image_standardization( image )


            #image_placeholder= tf.placeholder( tf.float32,
            #                                   shape = (self.config.image_height,
            #                                            self.config.image_width,
            #                                            self.config.image_channel ) )

            image.set_shape( [self.config.image_height,
                              self.config.image_width,
                              self.config.image_channel] )
            image_input = tf.expand_dims( image, axis = 0)
            #output = test_step( image_input, self.network )

            with tf.device( self.config.device_string ) as scope:          
                output = self.network( image_input, dropout_placeholder )

            #output_norm = tf.divide( output, tf.norm( output, ord=2, axis = 1, keep_dims = True ) )
            sess.run(tf.global_variables_initializer())

            ### savers
            saver = tf.train.Saver()

            if not (self.config.prior_weights is None):
                saver.restore(sess, self.model_file)


        return output, sess, image_file, dropout_placeholder, image_input



    def run_test(self, image_file ):

        
        score  = self.session.run( [self.output], feed_dict={self.image_file:image_file,\
                                                             self.dropout:1.0})
        #imf, inp = self.session.run( [self.image_file,  self.image], feed_dict={self.image_file:image_file} )

        #print imf
        #print np.max(inp)
        return score
