import cv2
import numpy as np

def process(filename):
  
  # Open file 
  cap = cv2.VideoCapture(filename)
 
  # If open failed...
  if cap.isOpened() == False: 
    println("Error opening " + filename)
    
  count = 0
 
  # While frames remain...
  while(cap.isOpened()):
  
    # Read frame
    success, frame = cap.read()
    if success == True:
 
      count++
  
      println( count )
 
    # Break the loop
    else:
    
      break
 
  # When everything done, release the video capture object
  cap.release()
 
  # Closes all the frames
  cv2.destroyAllWindows()
 
process('20160109_094636A.mp4')
