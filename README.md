# Viewpoint
The goal of this project is to develop a one dimensional "relative viewpoint inference" capability. Given three panoramic images taken at positions p0 <= p1 <= p2 along a line, we wish to return an estimate of the relative position of the center image r1=(p1-p0)/(p2-p0). Each panoramic image looks upward and provides a circular "fisheye" field of view covering roughly 220 degrees - roughly 20 degrees below the horizon in every direction. The camera has the same angle of rotation about the vertical axis for all three images.

If this capability is effective for one dimensional relative localization and can then be extended for two dimensional relative localization, it would enable precision optical navigation based on benchmark images.

## Approach
The project uses a deep convolutional neural network regression model operating directly on triplets of circular projections of the visual field to predict the relative position of the center image.

## Collect video
We mounted a [panoramic video camera](https://www.amazon.com/Andoer-Fisheye-Panorama-Activities-Camcorder/dp/B01JUFQMFW/ref=pd_sbs_421_16?_encoding=UTF8&pd_rd_i=B01JUFQMFW&pd_rd_r=KNNK4V678MFWVSHM7P1W&pd_rd_w=MjUET&pd_rd_wg=l9Yos&psc=1&refRID=KNNK4V678MFWVSHM7P1W) to the top of a car and drove around for about half an hour to gather raw video footage to train the system. The video is 2048 x 2048 pixels at 30 frames per second in MP4 format. The camera faces upward and maps a field of view approximately 220 degrees wide to a circle 2048 pixels in diameter on the imager.

## Transform video to training data
We used a python script to process the video forming training input as follows. One aproach to preprocessing the video is to use OpenCV and iterate through the video extracting frames. Another approach is to call FFMPEG via python and route the FFMPEG output to python via a pipe. Based on my prior experience, instllation of OpenCV on Ubuntu 16.04 is painful, requiring dozens of steps and lots of troubleshooting, rebuilding, patching, etc. Installation of FFMPEG by contrast takes only a few moments, so I opted for this approach. The preprocessor python script accepts the following arguments:

* inputfile - specifies the training video (MP4)
* inputresolution - specifies the training video resolution (2048)
* outputresolution - specifies the output frame resolution (512)
* span - specifies the number of frames between benchmark frames
* frequency - specifies the number of frames between training samples

The preprocessor iterates through the video decoding each frame at the output resolution. It stores each frame in a binary file as a numpy array with dimensions (res,res,3).

Preprocessor.py should run acceptably on a single core Ubuntu 16.04 Google Compute Engine with no special memory requirements. It requires python and ffmpeg.

...

Work in progress.

On Google Compute Engine launch a standard Ubuntu 16.04 instance and run:

sudo apt-get update
sudo apt-get upgrade
sudo-apt-get install ffmpeg

sudo apt-get install python-pip  
sudo pip install numpy

and maintains a FIFO holding "span" frames. At intervals specified by the "frequency" argument, the preprocessor assembles a training sample by concatenating the frames at the head of the FIFO, the frame at the tail of the FIFO, a frame drawn randomly from the FIFO, and the relative position of that calculated frame within the FIFO (if the frame is n frames from the end, its relative position is n/span). It saves the concatenated uncompressed resampled RGB frames and the relative position of the center frame as one training sample. The process continues until the entire video is processed and produces a single file with multiple training samples.
