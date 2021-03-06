""" Variational Auto-Encoder Example.
Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.
References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
import pickle

'''
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''

# load data
rows = []
columns = []
with open('saved_data' + os.sep + 'data_tower.pickle', 'rb') as handle:
    d = pickle.load(handle)
    serialised_segments = d['serialised_segments']
    rows = d['rows']
    columns = d['columns']

np.random.shuffle( serialised_segments )

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# __MAX__
# split in batches
batches_train = []
batches_test = []
tmp_counter = 1
batch_idx_start = 0
batch_idx_end = batch_idx_start + batch_size
while batch_idx_end < serialised_segments.shape[0]:
    # decide whether to put it in test or train
    if tmp_counter%10 == 0:
        batch_idx_end = batch_idx_start + 4
        batches_test.append( serialised_segments[ batch_idx_start:batch_idx_end,: ] )
        batch_idx_start += 4
        batch_idx_end = batch_idx_start + batch_size
    else:
        batch_idx_end = batch_idx_start + batch_size
        batches_train.append( serialised_segments[ batch_idx_start:batch_idx_end,: ] )
        batch_idx_start += batch_size
    tmp_counter += 1

# Network Parameters
image_dim = rows*columns # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    # 'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    # 'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Building the decoder
def conv_decoder(x):
    # TensorFlow Layers automatically create variables and calculate their
    # shape, based on the input.
    x = tf.layers.dense(x, units= (int(np.floor(rows/4))-1) * (int(np.floor(columns/4))-1) * 128)
    x = tf.nn.tanh(x)
    # Reshape to a 4-D array of images: (batch, height, width, channels)
    # New shape: (batch, 15, 15, 128)
    x = tf.reshape(x, shape=[-1, int(np.floor(rows/4))-1, int(np.floor(columns/4))-1, 128])
    # Deconvolution, image shape: (batch, 32, 32, 64)
    x = tf.layers.conv2d_transpose(x, 64, [4,4], strides=2)
    # Deconvolution, image shape: (batch, 65, 64, 1)
    x = tf.layers.conv2d_transpose(x, 1, [3,2], strides=2)
    # Apply sigmoid to clip values between 0 and 1
    x = tf.nn.sigmoid(x)
    # back to flat
    x = tf.reshape( x , [ -1, rows*columns] )
    return x

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps

# Building the decoder (with scope to re-use these layers later)
decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
# decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
# decoder = tf.nn.sigmoid(decoder)
decoder = conv_decoder(decoder)

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # __MAX__
    batch_idx = 0
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = batches_train[ batch_idx ]
        batch_idx += 1
        batch_idx = batch_idx%len(batches_train)

        # Train
        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

    # Testing
    # Generator takes noise as input
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    # Rebuild the decoder to create image from noise
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    # decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    # decoder = tf.nn.sigmoid(decoder)
    decoder = conv_decoder(decoder)

    # Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((rows * n, columns * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * rows:(n - i) * rows, j * columns:(j + 1) * columns] = \
            x_mean[0].reshape(rows, columns)

    print("Printing Images")
    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.savefig('figs/VAE_music.png', dpi=300); plt.clf()
    # plt.show()