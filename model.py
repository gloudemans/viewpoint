# The training data file contains raw XxY RGB video frames. If X and Y are 
# nominally 512 pixels. Each frame is about 750KB, and there are 30 frames per
# second 24MB per second, 1.5GB per minute, etc. 

import os
import numpy as np
import tensorflow as tf
import argparse
import sys
import tempfile

FLAGS = None

# Construct one bath of training data.
# Open the specified raw video file. Read frames with the specified
# x and y resolution into a FIFO holding 2*span frames. Select a random
# start position p0 up to span frames into the FIFO. Select a query frame 
# up to span frames from start frame. Select an end frame p2 = p0 + span.
# Compute the relative position of the query frame between the start and 
# end frames as r = (p1-p0)/(p2-p0). Collect frames p0, p1, and p2 as a
# training input sample, and r as a training output sample. Return count 
# sample pairs with shape (count, y, x, 9) and another having dimension (count,1).

def get_training_data(filename, x, y, span, count):

  fifolen = 2*span+1
  fifo = np.zeros( (fifolen,y,x,3), dtype=np.uint8)
  tensor = np.zeros( (count,y,x,9), dtype=np.uint8)
  target = np.zeros( (count,1), dtype=np.float32)

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
            n0 =  np.random.randint(0, span);
            n1 =  np.random.randint(0, span+1);
            p0 = (frame+n0) % fifolen
            p1 = (frame+n0+n1) % fifolen
            p2 = (frame+n0+span) % fifolen
            tensor[ia[sample],:,:,:] = np.dstack( (fifo[p0,:,:,:], fifo[p1,:,:,:], fifo[p2,:,:,:]) )
            target[ia[sample]] = n1/(span)
            timer -= interval
            sample += 1
            print(sample)
          timer += 1
        frame += 1

  return(tensor, target)

def deepnn(x):

  conv1 = tf.layers.conv2d(
    inputs = x,
    strides = (2,2),
    filters = 32,
    kernel_size = [5, 5],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  conv2 = tf.layers.conv2d(
    inputs = conv1,
    strides = (2,2),
    filters = 32,
    kernel_size = [5, 5],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  conv3 = tf.layers.conv2d(
    inputs = conv2,
    strides = (2,2),
    filters = 32,
    kernel_size = [5, 5],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  conv4 = tf.layers.conv2d(
    inputs = conv3,
    strides = (2,2),
    filters = 32,
    kernel_size = [5, 5],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  flat = tf.reshape(conv4, [-1, 32**3])

  dense = tf.layers.dense(
    inputs = flat, 
    units = 1024,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    activation = tf.nn.elu)

  y_conv = tf.layers.dense(
    inputs = dense,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    units = 1)

  return y_conv, keep_prob

def main(_):
  
  filename = 'data.rgb'
  x = 512
  y = 512
  span = 100
  batch = 1000
  minibatch = 50
  
  # Get batch of training data
  tensor, target = get_training_data(filename, x, y, span, batch)
  
  # Create IO placeholders
  x  = tf.placeholder(tf.float32, [None, x, y, 9])
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

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(batch//minibatch):
      
      print(i)
      
      k = (i*minibatch) % batch
      x_batch = tensor[k:k+minibatch]
      y_batch = target[k:k+minibatch]
      
      if i % 1 == 0:
        train_accuracy = mse.eval(feed_dict={
            x: x_batch, y_: y_batch, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        
      out = sess.run(y_conv, feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
      
      print(out[0:10])
      print(y_batch[0:10])
        
      train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: tensor, y_: target, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main)
