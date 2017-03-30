from __future__ import print_function

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

import numpy as np
import matplotlib.pyplot as plt



# Parameters
learning_rate = 0.01
training_epochs = 100
display_step = 1

# Training Data
# define system size
Lx, Ly = 20, 20
Ns = Lx*Ly

# define Ising model aprams
T=4.0 # temperature


# load data
train_X=np.loadtxt("mag_vs_T_L%i_T=%.2f.txt" %(Lx,T),delimiter=",",dtype=np.int)
train_Y=np.loadtxt("energies_vs_T_L%i_T=%.2f.txt" %(Lx,T),delimiter=",",dtype=np.int)

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float32,shape=(n_samples,Ns))
Y = tf.placeholder(tf.float32,shape=(n_samples,))

# Set model weights
J = tf.Variable( tf.truncated_normal(shape=(Ns,Ns) ), name="weight",dtype=tf.float32)
h = tf.Variable( tf.truncated_normal(shape=(Ns,) ), name="weight",dtype=tf.float32)

# Construct a linear model
pred = 0.5*tf.einsum('ai,ij,aj->a',X,J,X) + tf.einsum('ai,i->a',X,h)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c) )

            h_min=tf.reduce_min(h, reduction_indices=[0])
            h_max=tf.reduce_max(h, reduction_indices=[0])

            print(sess.run((h_min,h_max)))
