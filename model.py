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

  x_image = x

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable("W_conv1", [5, 5, 9, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - 512->256.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 32])
    b_conv2 = bias_variable("b_conv2", [32])
    h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling 256->128.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
    
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable("W_conv3", [5, 5, 32, 32])
    b_conv3 = bias_variable("b_conv3", [32])
    h_conv3 = tf.nn.elu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Second pooling 128->64.
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)
    
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable("W_conv4", [5, 5, 32, 32])
    b_conv4 = bias_variable("b_conv4", [32])
    h_conv4 = tf.nn.elu(conv2d(h_pool3, W_conv4) + b_conv4)

  # Second pooling 64->32
  with tf.name_scope('pool4'):
    h_pool4 = max_pool_2x2(h_conv4)
    
  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable("W_fc1", [32 * 32 * 32, 1024])
    b_fc1 = bias_variable("b_fc1", [1024])

    h_pool2_flat = tf.reshape(h_pool4, [-1, 32 * 32 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 1 scalar
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable("W_fc2", [1024, 1])
    b_fc2 = bias_variable("b_fc2", [1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

def bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())

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
