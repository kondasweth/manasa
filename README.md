## histogrm
1.Title and Description
Title: Visualizing Color Histogram of an Image
Description: This Python script reads an image, calculates its color histogram, and visualizes the histogram using matplotlib.

2.Installation
pip install numpy opencv-python matplotlib

3.Libraries
`numpy` as `np`: For numerical computing.
`cv2` as `cv`: OpenCV library for image processing.
`pyplot` module from `matplotlib`library for plotting graphs.

4.Usage
A histogram allows you to see the frequency distribution of a data set. It offers an “at a glance” picture of a distribution pattern, charted in specific categories.

5.Example
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('/home/swetha-konda/Downloads/parrot.jpeg')
cv.imwrite("/home/swetha-konda/Downloads/prt.jpeg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()

6.Explaination
Imports:
`numpy`: Renamed as `np`, it's used for numerical computing.
`cv2`: Renamed as `cv`, it's the OpenCV library used for image processing.
`pyplot` from `matplotlib`: Renamed as plt, used for plotting graphs.
Reading and Writing Image:
`img = cv.imread('/home/swetha-konda/Downloads/parrot.jpeg')`: Reads an image file named "parrot.jpeg" from the specified path.
`cv.imwrite("/home/swetha-konda/Downloads/prt.jpeg", img)`: Writes the same image to another file named "prt.jpeg" in the same directory.
Histogram Calculation and Plotting:
`color = ('b','g','r')`: Defines colors for plotting the histogram. 'b' for blue, 'g' for green, and 'r' for red.
`for i, col in enumerate(color)`: Iterates over each color channel in the image.
`histr = cv.calcHist([img], [i], None, [256], [0, 256])`: Calculates the histogram of the current color channel using OpenCV's calcHist function.
`plt.plot(histr, color=col)`: Plots the histogram using Matplotlib's plot function, with the specified color.
`plt.xlim([0, 256])`: Sets the x-axis limit of the plot from 0 to 256 (the range of pixel values).
Display Histogram:
`plt.show()`: Displays the histogram plot.
This code essentially reads an image, calculates and plots the histogram for each color channel (blue, green, and red), and displays the histograms using Matplotlib. It helps visualize the distribution of pixel intensities for each color channel in the image.
7.output&input
input                                        
![parrot](https://github.com/kondasweth/manasa/assets/169050846/782d7016-64c3-4641-a6a5-c81aac383da7)

ouput

![Figure_1](https://github.com/kondasweth/manasa/assets/169050846/da2aad9d-7524-4078-aefa-bea3cac3a93c) 


## web
1.Title and Description
Title: Webcam Video Recorder
Description: This Python script captures video from the webcam, converts the frames to RGB format, and saves them to a video file using OpenCV.

2.Installation
pip install opencv-python

3.Libraries
OpenCV(`cv2`):
`import cv2 as cv`: OpenCV is a library primarily aimed at real-time computer vision. It provides various functions and classes for image and video processing.

4.Usage
A webcam is a digital camera that captures video and audio data and transmits it in real-time over the internet. It is commonly used for video conferencing, live streaming, online meetings, and recording videos.

5.Example
import cv2 as cv
cam = cv.VideoCapture(0)
cc = cv.VideoWriter_fourcc(*'XVID')
file = cv.VideoWriter('output.avi', cc, 15.0, (640, 480))
if not cam.isOpened():
   print("error opening camera")
   exit()
while True:
   # Capture frame-by-frame
   ret, frame = cam.read()
   # if frame is read correctly ret is True
   if not ret:
      print("error in retrieving frame")
      break
   img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   cv.imshow('frame', img)
   file.write(img)

   
   if cv.waitKey(1) == ord('q'):
      break

cam.release()
file.release()
cv.destroyAllWindows()

6.Explaination
Opening the Webca
cam = cv.VideoCapture(0)
This line initializes a capture object to access the webcam.
`0` represents the index of the webcam (usually the default webcam).
Creating a VideoWriter Object
cc = cv.VideoWriter_fourcc(*'XVID')
file = cv.VideoWriter('output.avi', cc, 15.0, (640, 480))
This part initializes a VideoWriter object to write the captured frames to a video file.
VideoWriter_fourcc() specifies the codec for writing the video (in this case, XVID).
'output.avi' is the name of the output video file.
 15.0 is the frames per second (FPS) of the output video.
 (640, 480) is the resolution of the video (width, height).
Checking Camera Availability
if not cam.isOpened():
   print("error opening camera")
   exit()
 while True:
   ret, frame = cam.read()
   if not ret:
      print("error in retrieving frame")
      break
This checks if the webcam is successfully opened. If not, it prints an error message and exits the program.
Converting Color Format
img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
This line converts the color format of the captured frame from BGR to RGB.
Displaying Frames
cv.imshow('frame', img)
This line displays the captured frame using OpenCV's `imshow()` function.
Writing Frames to Video File
file.write(img)
This writes the converted frame to the output video file using the VideoWriter object.
Exiting the Program    
if cv.waitKey(1) == ord('q'):
   break
This checks for the 'q' keypress. If 'q' is pressed, it breaks out of the loop.
Releasing Resources
cam.release()
file.release()
cv.destroyAllWindows()
Finally, this releases the resources (webcam and video file) and closes all OpenCV windows.

7.input&output
input



 
    









