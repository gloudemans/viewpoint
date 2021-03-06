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

  training = True
  
  batch1 = tf.layers.batch_normalization(
    inputs = x,
    training = training)

  conv1 = tf.layers.conv2d(
    inputs = batch1,
    strides = (2,2),
    filters = 32,
    kernel_size = [11, 11],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  batch2 = tf.layers.batch_normalization(
    inputs = conv1,
    training = training)  
  
  conv2 = tf.layers.conv2d(
    inputs = batch2,
    strides = (2,2),
    filters = 32,
    kernel_size = [11, 11],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  batch3 = tf.layers.batch_normalization(
    inputs = conv2,
    training = training)
  
  conv3 = tf.layers.conv2d(
    inputs = batch3,
    strides = (2,2),
    filters = 32,
    kernel_size = [11, 11],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  batch4 = tf.layers.batch_normalization(
    inputs = conv3,
    training = training)
  
  conv4 = tf.layers.conv2d(
    inputs = batch4,
    strides = (2,2),
    filters = 32,
    kernel_size = [11, 11],
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    bias_initializer = tf.zeros_initializer(),
    padding = "same",
    activation = tf.nn.elu)
  
  flat = tf.reshape(conv4, [-1, 32*32*32])
  
  batch5 = tf.layers.batch_normalization(
    inputs = flat,
    training = training)

  dense1 = tf.layers.dense(
    inputs = batch5, 
    units = 1024,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    activation = tf.nn.elu)

  batch6 = tf.layers.batch_normalization(
    inputs = dense1,
    training = training)
  
  dense2 = tf.layers.dense(
    inputs = batch6, 
    units = 1024,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    activation = tf.nn.elu)
  
  batch7 = tf.layers.batch_normalization(
    inputs = dense2,
    training = training)
  
  y_conv = tf.layers.dense(
    inputs = batch7,
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
    units = 1)

  return y_conv

def main(_):
  
  rgb_file = '/home/mark/ssd/allvideo.rgb'
  npy_file = '/home/mark/ssd/allvideo100.npy'  
  xpix = 512
  ypix = 512
  span = 100
  batch = 5000
  minibatch = 100
  
  try:
    f = open(npy_file,'rb')
    x_batch = np.load(f);
    y_batch = np.load(f);
    f.close()
    print('Batch loaded')
  except:   
    # Get batch of training data
    x_batch, y_batch = get_training_data(rgb_file, xpix, ypix, span, batch)
    f = open(npy_file,'wb')
    np.save(f, x_batch);
    np.save(f, y_batch);
    f.close()
    print('Batch saved')
  
  # Create IO placeholders
  x  = tf.placeholder(tf.float32, [None, xpix, ypix, 9])
  y_ = tf.placeholder(tf.float32, [None, 1])

  # Build the graph for the deep net
  y_conv = deepnn(x)
  
  print('Graph complete')

  with tf.name_scope('loss'):
    mse = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    #train_op = train_step.minimize(mse)
    # with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-2).minimize(mse)

  #graph_location = tempfile.mkdtemp()
  #print('Saving graph to: %s' % graph_location)
  #train_writer = tf.summary.FileWriter(graph_location)
  #train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())    
    epoch = 0
    while True:    
      print('Epoch: ' + str(epoch))
      for i in range(batch//minibatch):
        print('Minibatch: ', i)
        k = (i*minibatch) % batch
        x_minibatch = x_batch[k:k+minibatch]
        y_minibatch = y_batch[k:k+minibatch]
        if i % 10 == 0:
          train_accuracy = mse.eval(feed_dict={
              x: x_minibatch, y_: y_minibatch})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: x_minibatch, y_: y_minibatch})    
      epoch += 1

if __name__ == '__main__':
  tf.app.run(main=main)
