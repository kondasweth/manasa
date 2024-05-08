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

7.output
[Screencast from 08-05-24 01:20:02 PM IST.webm](https://github.com/kondasweth/manasa/assets/169050846/6feb9aed-c1df-4ff4-8cb5-c91280695dbc)

## Bounding boxes
1.Title:Draw bounding boxes for images.

2.Installation
pip install pillow

3.Libraries
`os`: Provides functions for interacting with the operating system.
`csv`: Allows reading and writing CSV files.
`PIL.Image and PIL.ImageDraw: Modules from the Python Imaging Library (PIL)` used for image manipulation and drawing.

4.Usage
 Bounding boxes are used to label data for computer vision tasks, including: Object Detection: Bounding boxes identify and localize objects within an image, such as detecting pedestrians, cars, and animals. They represent object locations and are compatible with many machine-learning algorithms.

 5.Example
 import os
import csv
from PIL import Image, ImageDraw


csv_file = "/home/swetha-konda/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/swetha-konda/Downloads/7622202030987"
output_dir = "/home/swetha-konda/Downloads/7622202030987_with_boxes"


os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

6.Explaination
File Paths: The script specifies file paths for input CSV file (`csv_file`), directory containing images (`image_dir`), and output directory for images with bounding boxes (`output_dir`).
Create Output Directory: It creates the output directory (`output_dir`) if it doesn't already exist.
Functions:
draw_boxes(`image, boxes`): This function takes an image and a list of bounding box coordinates and draws red rectangles around objects defined by the bounding boxes.
crop_image(`image, boxes`): This function crops the image based on the bounding box coordinates and returns a list of cropped images.
Reading CSV File: The script opens the CSV file (`csv_file`) in read mode and reads its contents using csv.DictReader, which interprets each row as a dictionary where column headers are keys.
For each row in the CSV file:
It extracts the image filename from the 'filename' column.
Constructs the full path of the image.
Opens the image using Image.open() from the PIL library.
Extracts the bounding box coordinates from the CSV row and converts them into a list of dictionaries. Calls crop_image() to crop the image based on the bounding box coordinates.
Saves each cropped image with a prefix indicating its index and the original image filename. Draws bounding boxes on the original image using draw_boxes(). Saves the original image with bounding boxes.
Output: Cropped images are saved in the output directory with filenames prefixed by their index and the original image filename.
Images with bounding boxes drawn on them are saved in the output directory with filenames prefixed by "full_" and the original image filename.
This script seems designed for processing images with associated bounding box annotations, commonly used in object detection tasks.

7.Input&Output
input
![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/kondasweth/manasa/assets/169050846/9789d356-8f89-4c2a-b494-7e1fe0c2217f)
output
![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/kondasweth/manasa/assets/169050846/00132406-52b4-4751-a998-1e0c235c3799)
![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/kondasweth/manasa/assets/169050846/4ad3d256-7ff4-49a8-86f0-728670a0fe77)













 
    









