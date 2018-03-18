'''
implements the training and validation loops
'''



import tensorflow as tf

from tf_test.test_step import test_step


class Tester():

    def __init__(self, network, config):

        ## network definition comes from client
        ## makes this part independent
        self.network     = network
        self.config      = config
        #self.file_list   = self.config.file_list
        self.model_file  = self.config.test_prior_weights


        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True

        self.gpu_config = gpu_config
        self.output, self.session, self.image_place = self._prepare_graph_session()

    def _prepare_graph_session(self):

        g = tf.Graph()

        with g.as_default():

            sess = tf.Session(graph=g, config = self.gpu_config)

            image_placeholder= tf.placeholder( tf.float32,
                                               shape = (self.config.image_height,
                                                        self.config.image_width,
                                                        self.config.image_channel ) )
            image_input = tf.expand_dims( image_placeholder, axis = 0)
            output = test_step( image_input, self.network )
            sess.run(tf.global_variables_initializer())

            ### savers
            saver = tf.train.Saver()

            if not (self.config.prior_weights is None):
                saver.restore(sess, self.model_file)


        return output, sess, image_placeholder

    def run_test(self, image ):

        score  = self.session.run( [self.output], feed_dict={self.image_place:image })
        return score