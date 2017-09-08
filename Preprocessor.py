import cv2
import numpy as np

def process(filename):
  
  # Open file 
  cap = cv2.VideoCapture(filename)
 
  # If open failed...
  if cap.isOpened() == False: 
    println("Error opening " + filename)
 
  # While frames remain...
  while(cap.isOpened()):
  
    # Read frame
    success, frame = cap.read()
    if success == True:
 
      # Display the resulting frame
      cv2.imshow('Frame', frame)
 
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
    # Break the loop
    else:
    
      break
 
  # When everything done, release the video capture object
  cap.release()
 
  # Closes all the frames
  cv2.destroyAllWindows()
