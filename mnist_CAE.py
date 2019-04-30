""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
rows = 28
columns = 28

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
    x = tf.layers.dense(x, 4608)
    x = tf.nn.tanh(x)
    return x


# Building the decoder
def decoder(x):
    # TensorFlow Layers automatically create variables and calculate their
    # shape, based on the input.
    x = tf.layers.dense(x, units=6 * 6 * 128)
    x = tf.nn.tanh(x)
    # Reshape to a 4-D array of images: (batch, height, width, channels)
    # New shape: (batch, 6, 6, 128)
    x = tf.reshape(x, shape=[-1, 6, 6, 128])
    # Deconvolution, image shape: (batch, 14, 14, 64)
    x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
    # Deconvolution, image shape: (batch, 28, 28, 1)
    x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
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

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x.reshape([rows, columns, 1])})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j]

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.savefig('figs/mnist_test_original.png', dpi=300); plt.clf()
    # plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig('figs/mnist_test_reconstructed.png', dpi=300); plt.clf()
    # plt.show()