import os
import numpy as np

def get_training_data(filename, x, y, span, count):

    fifo = np.zeros( (span+1,y,x,3), dtype=np.uint8)
    tensor = np.zeros( (count,y,x,3), dtype=np.uint8)
    target = np.zeros( (count), dtype=np.float32)
   
    frame_length = 3*x*y
    frames = os.stat(filename).st_size/length
    interval = (frames-(span+1))/count
    timer = 0
    sample = 0

    with open(filename) as f:
    
        frame = 0
        while True:
            raw_image = f.read(length)
            if len(raw_image) < length:
                break
            else:
                image = np.fromstring(raw_image, dtype='uint8')
                fifo[frame % (span+1), :,:,:] = np.reshape(image, (y,x,3))
                if frame > span:
                    if timer >= 0:
                        n =  np.random.randint(0, span+1);
                        p0 = (frame+0) % (span+1)
                        p1 = (frame+n) % (span+1)
                        p2 = (frame+span) % (span+1)
                        tensor[sample,:,:,:] = np.dstack( (fifo[p0,:,:,:], fifo[p1,:,:,:], fifo[p2,:,:,:]) )
                        target[sample] = n/(span+1)
                        timer = timer - interval
                        sample += 1
                    timer += 1
                frame += 1
    return(tensor, target)
