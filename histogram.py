import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help = "Enter the path of the image")

parser.add_argument("out_dir", help = "name of the output directory where you want to save your output")

args = parser.parse_args()
image_dir = args.image_path
 
img = cv.imread('/home/swetha-konda/Downloads/parrot.jpeg')
cv.imwrite("/home/swetha-konda/Downloads/prt.jpeg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
