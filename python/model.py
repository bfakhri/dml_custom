# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np


# Constants to eventually parameterise
## Base Dir to write logs to
BASE_LOGDIR = './logs/'
## Subdirectory for this experiment
RUN = '3'
## Learning Rate for Adam Optimizer
LEARN_RATE = 1e-4
## Number of images to push through the network at a time
BATCH_SIZE = 256 
## Number of Epochs to train for
MAX_EPOCHS = 10 
## How many training steps between outputs to screen and tensorboard
output_steps = 20
## Enable or disable GPU (0 disables GPU, 1 enables GPU)
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})

# Define functions that create useful variables
def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="B"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 2D Convolution Func 
def conv2d(x, W, name='conv'):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max-Pooling Function - Pooling explained here: 
# http://ufldl.stanford.edu/tutorial/supervised/Pooling/
def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define a Convolutional Layer 
def conv_layer(x, fan_in, fan_out, name="convl"):
    with tf.name_scope(name):
        # Create Weight Variables
        W = weight_variable([5, 5, fan_in, fan_out], name="W")
        B = bias_variable([fan_out], name="B")
        # Convolve the input using the weights
        conv = conv2d(x, W)
        # Push input+bias through activation function
        activ = tf.nn.relu(conv + B)
        # Create histograms for visualization
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", B)
        tf.summary.histogram("Activations", activ) 
        # MaxPool Output
        return max_pool_2x2(activ)


class Model:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        # Begin Defining the Computational Graph
        with tf.name_scope('MainGraph'):
            with tf.name_scope('Inputs'):
                # Placeholders for data and labels
                self.input = tf.placeholder(tf.float32, shape=input_shape)
                tf.summary.image('sample_image', self.input, 3)

            # Convolution Layers
            conv1 = conv_layer(self.input, 3, 32, name='Conv1') 
            conv2 = conv_layer(conv1, 32, 64, name='Conv2') 
            
            # Here we implement spatial softmax
            conv2_exp = conv2
            features = tf.reshape(tf.transpose(conv2_exp, [0, 3, 1, 2]), [-1, conv2.shape[1]*conv2.shape[2]])
            conv2_ssm = tf.nn.softmax(features)
            # Reshape and transpose back to original format.

            self.conv2_ssm = tf.transpose(tf.reshape(conv2_ssm, [-1, 1, conv2.shape[1], conv2.shape[2]]), [0, 2, 3, 1])
            print(self.conv2_ssm.shape)
            #conv2_ssm =  tf.contrib.layers.spatial_softmax(conv2, name='Conv2-ssm') 
            #tf.summary.image('conv2_viz_smm1', tf.expand_dims(conv2_ssm[:,:,:,0], axis=3), 3)
            tf.summary.image('conv2_viz_smm1', self.conv2_ssm[:,:,:], 3)
            #tf.summary.image('conv2_viz_smm2', tf.expand_dims(conv2_ssm[:,:,:,0], axis=3), 3)

            # Fully Connected Layers
            with tf.name_scope('FC1'):
                h_flat = tf.layers.flatten(conv2_ssm)
                W_fc1 = weight_variable([conv2_ssm.shape[1:].num_elements(), self.input.shape[1:].num_elements()])
                b_fc1 = bias_variable([self.input.shape[1:].num_elements()])
                
                self.input_hat = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
                self.input_hat = tf.reshape(self.input_hat, shape=(-1,)+self.input_shape[1:])


            with tf.name_scope('Objective'):
                # Define the objective function
                self.mse = tf.losses.mean_squared_error(labels=self.input, predictions=self.input_hat)
                tf.summary.scalar('mse', self.mse)


        # Define the training step
        self.train = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.mse)

        # Create the session
        self.sess = tf.Session(config=SESS_CONFIG)

        # Init all weights
        self.sess.run(tf.global_variables_initializer())

        # Merge Summaries and Create Summary Writer for TB
        self.all_summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(BASE_LOGDIR + RUN)
        self.writer.add_graph(self.sess.graph) 

    def train_step(self, batch_x):
        with self.sess.as_default():
            all_sums_out, mse_out, ssm_out = self.sess.run([self.all_summaries, self.mse, self.conv2_ssm], feed_dict={self.input: batch_x})
            print('Step: ' + str(cur_step) + '\t\tMSE: ' + str(train_accuracy))
            print(sum(sum(sum(ssm_out))))
            self.writer.add_summary(all_sums_out, cur_step) 
            self.train.run(feed_dict={x: batch_x})



