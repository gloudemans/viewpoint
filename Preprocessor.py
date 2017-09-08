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
  
  pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=6*10*res^2)
  
  length = 3*res^2
  
  frame = 0
  while True:
    raw_image = pipe.stdout.read(length)
    if len(raw_image) < length:
      break
    else:
      image = np.fromstring(raw_image, dtype='uint8')
      image = np.reshape((res,res,3))
      frame += 1
      print(frame)

process('20160109_094636A.mp4', 512, 100, 100)
