# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:31 2018

@author: kgicmd
"""

import os
os.getcwd()
os.chdir("C:\\Users\\kgicmd\\proj2")

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# batch size
batch_size = 100

# number of batches
num_batch = mnist.train.num_examples // batch_size

# place_holder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# dropout
prob = tf.placeholder(tf.float32)

# set up a sinple neural network
#w = tf.Variable(tf.zeros((784,10)))
#b = tf.Variable(tf.zeros((10)))
#pred = tf.nn.softmax(tf.matmul(x,w)+b)

w1 = tf.Variable(tf.truncated_normal([784,2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
l1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
l1_drop = tf.nn.dropout(l1, keep_prob=prob)

w2 = tf.Variable(tf.truncated_normal([2000,2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
l2 = tf.nn.tanh(tf.matmul(l1_drop, w2) + b2)
l2_drop = tf.nn.dropout(l2, keep_prob=prob)

w3 = tf.Variable(tf.truncated_normal([2000,1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
l3 = tf.nn.tanh(tf.matmul(l2_drop, w3) + b3)
l3_drop = tf.nn.dropout(l3, keep_prob=prob)

w4 = tf.Variable(tf.truncated_normal([1000,10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
pred = tf.nn.softmax(tf.matmul(l3, w4) + b4)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

# gradient descent
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step=tf.train.AdadeltaOptimizer(0.01).minimize(loss)

# initialize
init = tf.global_variables_initializer()

# accuracy
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y, prob:0.95})
            
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,prob:0.95})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images,y:mnist.train.labels,prob:0.95})
        print("Epoch:" + str(epoch) + " Testing Accuracy: " + str(test_acc) +\
              " Training Accuracy: " + str(train_acc))
