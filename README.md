<<<<<<< HEAD
## histogrm
1.Title: and Description
Title: Visualizing Color Histogram of an Image
Description: This Python script reads an image, calculates its color histogram, and visualizes the histogram using matplotlib.

2.Installation:
pip install numpy opencv-python matplotlib

3.Libraries:
`numpy` as `np`: For numerical computing.
`cv2` as `cv`: OpenCV library for image processing.
`pyplot` module from `matplotlib`library for plotting graphs.

4.Usage:
A histogram allows you to see the frequency distribution of a data set. It offers an “at a glance” picture of a distribution pattern, charted in specific categories.

5.Example:
=======
##histogram

1.install following packages
```numpy, opencv, matplootlib```

2.code
```bash


>>>>>>> aaa4cfb... add new file
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
<<<<<<< HEAD

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


## webcam
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

5.Example:
import cv2 

video = cv2.VideoCapture(0) 
 
if (video.isOpened() == False):  
    print("Error reading video file") 

frame_width = int(video.get(3)) 
frame_height = int(video.get(4)) 
   
size = (frame_width, frame_height) 
   

result = cv2.VideoWriter('camera.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
    
while(True): 
    ret, frame = video.read() 
  
    if ret == True:  

        result.write(frame) 
        cv2.imshow('Frame', frame) 
  
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
  
    else: 
        break
  
video.release() 
result.release() 
    
cv2.destroyAllWindows() 
   
print("The video was successfully saved")

6.Explaination
Importing Libraries: The code begins by importing the OpenCV library with the alias `cv2`.

Opening the Camera: It creates a video capture object video using `cv2`.`VideoCapture(0)`, which opens the default camera (usually the webcam).

Checking Camera Status: It checks if the camera is opened successfully using `video.isOpened()`. If the camera fails to open, it prints an error message.
Creating VideoWriter Object: It creates a VideoWriter object named result to save the video to a file named "camera.avi". It specifies the codec (MJPG), frames per second (fps), and size of the video frames.
Capturing and Saving Video: Inside a `while loop`, it continuously captures frames from the camera using `video.read()`. If a frame is read successfully `(ret == True)`, it writes the frame to the video file using `result.write(frame)` and displays it using `cv2.imshow()`. Pressing 'S' on the keyboard stops the process (`cv2.waitKey(1) & 0xFF == ord('s'))`.
Releasing Resources: After exiting the loop, it releases the video capture and video write objects using `video.release()` and `result.release()` respectively.
Closing Windows: It closes all the OpenCV windows using `cv2.destroyAllWindows()`.
Print Success Message: Finally, it prints a success message indicating that the video was saved successfully.

7.output
https://github.com/kondasweth/manasa/assets/169050846/6b871eb9-12a4-4e3a-b35f-9d330e61d448

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

## number
1.Title:iteration (this will be explain each iteration,print the sum of the current and previous numbers)

2.Example
def print_sum_of_current_and_previous():
    previous_number = 0
    for i in range(1, 11):
        current_number = i
        sum_of_previous_and_current = previous_number + current_number
        print(f"Current number: {current_number}, Previous number: {previous_number}, Sum: {sum_of_previous_and_current}")
        previous_number = current_number

print_sum_of_current_and_previous()

3.Explaination
Function Definition:
def print_sum_of_current_and_previous():
This line defines a function named `print_sum_of_current_and_previous()`. It doesn't take any arguments.
Initialization:
previous_number = 0
It initializes a variable `previous_number` to 0. This variable will store the previous number in each iteration.
Loop:
for i in range(1, 11):
It assigns the current value of the loop variable `i` to `current_number`.
sum_of_previous_and_current = previous_number + current_number
It calculates the sum of the previous number and the current number and stores it in
`sum_of_previous_and_current`
print(f"Current number: {current_number}, Previous number: {previous_number}, Sum: {sum_of_previous_and_current}")
It prints the current number, the previous number, and their sum.
previous_number = current_number
It updates `previous_number` to the current number for the next iteration.
Function Call:
print_sum_of_current_and_previous()
It calls the function print_sum_of_current_and_previous().
The output of running this code will be a series of lines printed to the console, each displaying the current number, the previous number, and their sum, for the numbers 1 to 10. In each line, the current number will be the value of i, the previous number will be the value of i - 1, and the sum will be the value of i + (i - 1).

Output:
Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17














 
    









=======
```
>>>>>>> aaa4cfb... add new file
