#!/usr/bin/env python
#-------------------
# Tensor Flow 1'St try
# Task: MNIST Classification
# Model: Soft Max Regression
# Function: y = softmax(Wx + b)
# @ CityU HK, Zhang Hao
#-------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) 
# x isn't a specific value, it's a placeholder, a value we'll input when ask TensorFolder to run computation, Each  data is 784-Dim, here None means that it can be any value

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# We initialize both "W" and "b" as tensors full of zeros.

y = tf.nn.softmax(tf.matmul(x, W) + b) # This is our model
# 1'St, we multiply x by W with expression tf.matmul(x,W). This is flipped from when we multiplied them in our equation, where we had Wx as a small trick to deal with x being a 2D tensor with multiple inputs.
# We then add b, and finally applty tf.nn.softmax

y_ = tf.placeholder(tf.float32, [None, 10])
# Lablels for each sample xi

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Cross Entropy loss

####
# Train Procedure
####
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

####
# Evaluation Procedure
####
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
