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
To use this script, replace the path in 'cv.imread()' with the path to your image file. Then, run the script:

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
7.output
input                                        
file:///home/swetha-konda/Downloads/parrot.jpeg 
ouput
file:///home/swetha-konda/Desktop/code1/Figure_1.png



