import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import Dataset
from tensorflow.contrib.data import Iterator

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
                        tensor[ia(sample),:,:,:] = np.dstack( (fifo[p0,:,:,:], fifo[p1,:,:,:], fifo[p2,:,:,:]) )
                        target[ia(sample)] = n/(span+1)
                        timer -= interval
                        sample += 1
                        print(sample)
                    timer += 1
                frame += 1

    return(tensor, target)

# The file data.rgb contains raw video frames
# 30 minutes of 512x512 RGB frames at 30 frames per second is about 42GB.
# For training we can traverse this file and construct a numpy array of
# of 1000 512x512x9 frames and then shuffle it.




# Prepare training data
# Train
# Prepare validation data
# Validate



 Iterator

# Toy data
train_imgs = tf.constant(['train/img1.png', 'train/img2.png',
                          'train/img3.png', 'train/img4.png',
                          'train/img5.png', 'train/img6.png'])
train_labels = tf.constant([0, 0, 0, 1, 1, 1])

val_imgs = tf.constant(['val/img1.png', 'val/img2.png',
                        'val/img3.png', 'val/img4.png'])
val_labels = tf.constant([0, 0, 1, 1])

# create TensorFlow Dataset objects
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

# create TensorFlow Iterator object
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
