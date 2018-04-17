'''
implements the training and validation loops
'''

import numpy as np

import tensorflow as tf

from data_input_pipeline.data_preprocess import prepare_dataset
from train_step import train_step, \
    prep_train_summary_op, \
    prep_valid_summary_op

class Trainer():

    def __init__(self, network, config):

        ## network definition comes from client
        ## makes this part independent
        self.network     = network
        self.config      = config
        ## tf record parser object with parse_example signature
        ## the parser is slightly different for training and
        ## validation dataset because the former may contain
        ## some augmentation
        ## these parsers are supplied by client
        ##TODO: how to make client define a signature function???
        self.train_parser = config.train_parser
        self.valid_parser = config.valid_parser
        ## the tfrecords files for training and validation
        self.train_tfrecords = config.train_tfrecords
        self.valid_tfrecords = config.valid_tfrecords

        self.contrastive_margin = config.contrastive_margin

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True

        self.gpu_config = gpu_config

    def train(self):

        g = tf.Graph()

        with g.as_default():

            sess = tf.Session(graph=g, config = self.gpu_config)

            training_dataset = prepare_dataset(self.train_tfrecords,
                                               self.train_parser.parse_example,
                                               self.config.batch_size).repeat()

            validation_dataset = prepare_dataset(self.valid_tfrecords,
                                                 self.valid_parser.parse_example,
                                                 self.config.batch_size)

            ## from TF 1.4: this is a way to reuse one iterator
            ## for multiple datasets
            handle = tf.placeholder(tf.string, shape=[])
            data_iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
            next_element = data_iterator.get_next()


            ##the train data iterator needs to be initialized one
            ## because it is infinite (.repeat()
            ## validation data iterator is initializable multiple times
            train_data_iterator = training_dataset.make_one_shot_iterator()
            valid_data_iterator = validation_dataset.make_initializable_iterator()

            ## computation graph - using placeholders
            ## TODO: placeholder is slow; check if any other faster method
            ## works with the reusable iterators
            image_placeholder_one= tf.placeholder( tf.float32,
                                               shape = (None, self.config.image_height,
                                                        self.config.image_width,
                                                        self.config.image_channel ) )
            image_placeholder_two = tf.placeholder(tf.float32,
                                                   shape=(None, self.config.image_height,
                                                          self.config.image_width,
                                                          self.config.image_channel))

            label_placeholder = tf.placeholder(tf.int64,
                                               shape=(None, ))

            dropout_placeholder = tf.placeholder( tf.float32, shape =  () )

            output_one, output_two, loss, learning_rate, \
            global_steps, train_op = \
                train_step(images_one = image_placeholder_one,
                           images_two=image_placeholder_two,
                           labels = label_placeholder,
                           network = self.network,
                           dropout_prob = dropout_placeholder,
                           learning_rate_info=self.config.learning_rate_info,
                           device_string=self.config.device_string,
                           loss_op = self.config.loss_op,
                           one_hot = self.config.one_hot,
                           loss_op_kwargs = self.config.loss_op_kwargs,
                           margin = self.contrastive_margin ,
                           optimizer=self.config.optimizer,
                           optimizer_kwargs=self.config.optimizer_kwargs)

            ### summary_ops
            # easy way, create sum place holders and create the summary op

            total_loss_summ = tf.placeholder( tf.float32, shape=[] )
            learn_rate_summ  = tf.placeholder( tf.float32, shape = [] )


            train_summary_op = prep_train_summary_op( total_loss_summ,
                                                 learn_rate_summ)
            valid_summary_op = prep_valid_summary_op( total_loss_summ )

            ###

            ### savers
            saver = tf.train.Saver( max_to_keep=20,
                                    pad_step_number= True,
                                    keep_checkpoint_every_n_hours = 1)



            train_summary_writer = tf.summary.FileWriter( logdir=\
                                                self.config.train_summary_path)
            valid_summary_writer = tf.summary.FileWriter( logdir = \
                                                self.config.valid_summary_path )

            ##save the graph
            train_summary_writer.add_graph(g)

            ###


            sess.run(tf.global_variables_initializer())
            train_handle = sess.run(train_data_iterator.string_handle())
            valid_handle = sess.run( valid_data_iterator.string_handle() )

            if not (self.config.prior_weights is None):
                saver.restore(sess, self.config.prior_weights)
                current_step, l_rate = sess.run([global_steps, learning_rate])
                print 'Starting from Step {0} and learning rate {1}'.format( \
                                current_step, l_rate )

            ## BIG Loop
            while True:

                ##main training loop
                print 'Entering training loop...'
                for _ in range( self.config.batch_per_epoch ):

                    images_one, images_two, labels, files_one, files_two = \
                        sess.run( next_element ,feed_dict={handle:train_handle})
                    _,out_1,out_2,loss_output = sess.run( [train_op,output_one, output_two,loss],
                                                     feed_dict={image_placeholder_one:images_one,
                                                                image_placeholder_two: images_two,
                                                                label_placeholder:labels,
                                                                dropout_placeholder:0.5} )

                    if np.isnan( loss_output ):
                        print files_one, files_two
                        print 'Loss: {0}'.format(loss_output)
                        exit

                ##stop the training and collect some statistics
                current_step, l_rate = sess.run( [global_steps ,learning_rate] )
                print 'Step no {0}'.format( current_step )
                print 'Learning rate {0}'.format( l_rate )

                ### collecting training stats
                print 'Start collecting training statistics...'
                t_loss = 0.
                for _ in range(self.config.batch_per_test):
                    images_one, images_two, labels,files_one,files_two = sess.run(next_element, feed_dict={handle: train_handle})


                    loss_output= sess.run(loss, \
                                          feed_dict={image_placeholder_one: images_one,
                                                     image_placeholder_two: images_two,
                                                     label_placeholder: labels,
                                                     dropout_placeholder:1.0})

                    if np.isnan( loss_output ):
                        print files_one, files_two
                        print 'Loss: {0}'.format(loss_output)
                        exit

                    t_loss += loss_output

                t_loss /= np.float32( self.config.batch_per_test )

                print 'Saving Epoch Train Loss {0}'.format( t_loss )

                train_summary = sess.run( train_summary_op,
                                          feed_dict={ total_loss_summ:t_loss,
                                                      learn_rate_summ:l_rate })

                train_summary_writer.add_summary( train_summary, global_step=current_step)
                train_summary_writer.flush()

                ## collecting validation stats
                print 'Start collecting validation statisitics...'
                sess.run(valid_data_iterator.initializer)
                t_loss = 0.
                for _ in range(self.config.batch_per_test):
                    images_one, images_two, labels,files_one,files_two = sess.run(next_element, feed_dict={handle: valid_handle})

                    loss_output = sess.run(loss, \
                                           feed_dict={image_placeholder_one: images_one,
                                                      image_placeholder_two: images_two,
                                                      label_placeholder: labels,
                                                      dropout_placeholder:1.0})
                    if np.isnan( loss_output ):
                        print files_one, files_two
                        print 'Loss: {0}'.format(loss_output)
                        exit

                    t_loss += loss_output

                t_loss /= np.float32(self.config.batch_per_test)
                print 'Saving Epoch Validation Loss {0}'.format( t_loss )


                valid_summary = sess.run(valid_summary_op,
                                         feed_dict={total_loss_summ: t_loss})

                valid_summary_writer.add_summary(valid_summary, global_step=current_step)


                valid_summary_writer.flush()

                ## TODO: save the model
                ## possible configs : every time
                ## only if validation loss is decreasing

                saver.save( sess, self.config.model_checkpoint_path,\
                            global_step=current_step)
