# Viewpoint
The goal of this project is to develop a one dimensional "relative viewpoint inference" capability. Given three panoramic images taken at positions p0 <= p1 <= p2 along a line, we wish to return an estimate of the relative position of the center image r1=(p1-p0)/(p2-p0). Each panoramic image contains azimuth angles from 0 to 360 degrees and elevation angles from -20 to +20 degrees relative to the horizon.

## Approach
The project uses deep a convolutional neural network regression model operating directly on triplets of circular projections of the visual field.

## Collect raw video
We mounted a panoramic video camera to the top of a car and drove around for about half an hour to gather raw video footage to train the system. The video is 2048 x 2048 pixels at 30 frame per second in MP4 format. The camera faces upward and maps a field of view approximately 220 degrees wide to a circle 2048 pixels in diameter on the imager. This allows the camera to see elevation angles down to 20 degrees below the horizon in every direction.

## Transform video to training data
We used a python script to process the video forming training input as follows. One aproach to preprocessing the video is to use OpenCV and iterate through the video extracting frames. Another approach is to call FFMPEG via python and route the FFMPEG output to python via a pipe. Based on my prior experience, instllation of OpenCV on Ubuntu 16.04 is painful, requiring dozens of steps and lots of troubleshooting, rebuilding, patching, etc. Installation of FFMPEG by contrast takes only a few moments, so I opted for this approach. The preprocessor python script accepts the following arguments:

* inputfile - specifies the training video (MP4)
* inputresolution - specifies the training video resolution (2048)
* outputresolution - specifies the output frame resolution (512)
* span - specifies the number of frames between benchmark frames
* frequency - specifies the number of frames between training samples

The preprocessor iterates through the video decoding each frame at the output resolution and maintains a FIFO holding "span" frames. At intervals specified by the "frequency" argument, the preprocessor assembles a training sample by concatenating thre frame at the head of the FIFO, the frame at the tail of the FIFO, a frame drawn randomly from the FIFO, and the relative position of that calculated frame within the FIFO (if the frame is n frames from the end, its relative position is n/span). It saves the concatenated uncompressed resampled RGB frames and the relative position of the center frame as one training sample. The process continues until the entire video is processed and produces a single file with multiple training samples.

Run Preprocessor.py runs well on a single core Ubuntu 16.04 Google Compute Engine with no special memory requirements and ffmpeg installed.



