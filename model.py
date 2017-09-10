# The training data file contains raw video frames
# 30 minutes of 512x512 RGB frames at 30 frames per second is about 42GB.
# For training we can traverse this file and construct a numpy array of
# of 1000 512x512x9 frames and then shuffle it.

import os
import numpy as np
import tensorflow as tf
import argparse
import sys
import tempfile

FLAGS = None

def get_training_data(filename, x, y, span, count):

  fifo = np.zeros( (span+1,y,x,3), dtype=np.uint8)
  tensor = np.zeros( (count,y,x,9), dtype=np.uint8)
  target = np.zeros( (count), dtype=np.float32)

  length = 3*x*y
  frames = os.stat(filename).st_size/length
  interval = (frames-(span+1))/count
  timer = 0
  sample = 0

  ia = np.random.permutation(count)

  with open(filename,'rb') as f:

    frame = 0
    while True:
      raw_image = f.read(length)
      if len(raw_image) < length:
        break
      else:
        image = np.fromstring(raw_image, dtype='uint8')
        fifo[frame % (span+1), :,:,:] = np.reshape(image, (y,x,3))
        if frame > span:
          while timer >= 0:
            n =  np.random.randint(0, span+1);
            p0 = (frame+0) % (span+1)
            p1 = (frame+n) % (span+1)
            p2 = (frame+span) % (span+1)
            tensor[ia[sample],:,:,:] = np.dstack( (fifo[p0,:,:,:], fifo[p1,:,:,:], fifo[p2,:,:,:]) )
            target[ia[sample]] = n/(span+1)
            timer -= interval
            sample += 1
            print(sample)
          timer += 1
        frame += 1

  return(tensor, target)

def deepnn(x):

  x_image = x

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 9, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([128 * 128 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 128*128*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 1 scalar
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  
  filename = 'data.rgb'
  x = 512
  y = 512
  span = 100
  count = 10
  
  # Import data
  tensor, target = get_training_data(filename, x, y, span, count)
  #tensor = tf.constant(tensor, tf.float32)
  #target = tf.constant(target, tf.float32)
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 512, 512, 9])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    mse = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  print('A')
  with tf.Session() as sess:
    print('B')
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      print(i)
      k = (i*50) % count
      x_batch = tensor[k:k+50,:,:,:]
      y_batch = target[k:k+50,:]
      #if i % 100 == 0:
      #  train_accuracy = accuracy.eval(feed_dict={
      #      x: x_batch[0], y_: y_batch, keep_prob: 1.0})
      #  print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: tensor, y_: target, keep_prob: 1.0}))
  print('C')

if __name__ == '__main__':
  tf.app.run(main=main)
