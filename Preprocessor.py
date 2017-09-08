# Inspired by:
# http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/

import numpy as np
import subprocess as sp

def process(file, res, span, frequency):

  command = [ 'ffmpeg',
              '-i', file,
              '-vf', 'scale={0}:{0}'.format(res),
              '-f', 'image2pipe',
              '-pix_fmt', 'rgb24',
              '-vcodec', 'rawvideo', '-']
  
  pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=6*10*res**2)
  
  length = 3*res**2
  
  fifo = np.zeros( (span+1,res,res,3), dtype=np.uint8)
  
  f = open(file[:-3]+'vec', 'wb')
  
  frame = 0
  while True:
    raw_image = pipe.stdout.read(length)
    if len(raw_image) < length:
      break
    else:
      image = np.fromstring(raw_image, dtype='uint8')
      fifo[frame % (span+1), :,:,:] = np.reshape(image, (res,res,3))
      if frame and (frame%frequency)==0:
        n =  np.random.randint(0, span+1);
        p0 = (frame+0) % (span+1)
        p1 = (frame+n) % (span+1)
        p2 = (frame+span) % (span+1)
        tensor = np.dstack( (fifo[p0,:,:,:], fifo[p1,:,:,:], fifo[p2,:,:,:]) )
        target = n/(span+1)
        np.save(f, tensor)
        np.save(f, target)
        print('Vector ' + str(target))
      frame += 1
      print(frame)
    
  f.close()

process('20160109_094636A.mp4', 512, 100, 3)
