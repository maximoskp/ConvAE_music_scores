""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress 4-bar scores with
16ths resolution to
lower latent space and then reconstruct them.

Author: Maximos Kaliakatsos Papakostas, based on turial by Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
# __MAX__
# from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# __MAX__
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

# Training Parameters
learning_rate = 0.01 # it was 0.01
num_steps = 1000 # possibly increase a lot
batch_size = 256

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

display_step = 1000
examples_to_show = 10
num_input = rows*columns

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, rows, columns, 1])

# Building the encoder
def encoder(x):
    # Typical convolutional neural network to classify images.
    x = tf.layers.conv2d(x, 64, 5)
    x = tf.nn.tanh(x)
    x = tf.layers.average_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, 128, 5)
    x = tf.nn.tanh(x)
    x = tf.layers.average_pooling2d(x, 2, 2)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, 1024)
    x = tf.nn.tanh(x)
    # Output 2 classes: Real and Fake images
    x = tf.layers.dense(x, 2)
    return x


# Building the decoder
def decoder(x):
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
    return x

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, rows, columns, 1])

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((rows * n, columns * n))
    canvas_recon = np.empty((rows * n, columns * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        batch_x_reshaped = np.reshape(batch_x, newshape=[-1, rows, columns, 1])
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x_reshaped})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * rows:(i + 1) * rows, j * columns:(j + 1) * columns] = \
                batch_x[j].reshape([rows, columns])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * rows:(i + 1) * rows, j * columns:(j + 1) * columns] = \
                g[j].reshape([rows, columns])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.savefig('figs/music_CAE_test_original.png', dpi=300); plt.clf()
    # plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig('figs/music_CAE_test_reconstructed.png', dpi=300); plt.clf()
    # plt.show()