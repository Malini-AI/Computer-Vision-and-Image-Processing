import cv2
import numpy as np
import math
from PIL import Image
from PIL import GifImagePlugin

# vertical edge detection
image = cv2.imread("h.jpg",0)
avg = 0
# sobel Filter for x (vertical edge detection)
sobel_x = []
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

row = image.shape[0] #rows
col = image.shape[1] #colms

sobel_imgx= np.zeros(image.shape)

for i in range(1, row-1): # (row-2 and col-2 )
    for j in range(1, col-1):
        Sum = 0
        for m in range(3): #(filter 3x3)
            for n in range(3):
                Sum = Sum + (sobel_x[m][n] * image[i-1+m][j-1+n])
               
            sobel_imgx[i][j] = (Sum/200)

cv2.imshow("Sobel along - x", sobel_imgx)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Horizontal edge detection
image = cv2.imread("h.jpg",0)
#blur = cv2.blur(image,(1,1))
# sobel Filter for y (horizontal edge detection)
sobel_y = []
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

row = image.shape[0] #rows
col = image.shape[1] #colms

sobel_imgy= np.zeros(image.shape)

for i in range(row-1):
    for j in range(col-1):
        Sum = 0
        for m in range(3): #(filter 3x3)
            for n in range(3):
                Sum = Sum + (sobel_y[m][n] * image[i+m-1][j+n-1])
            
            sobel_imgy[i][j] = (Sum/200)

cv2.imshow("Sobel along - y", sobel_imgy)
cv2.waitKey(0)
cv2.destroyAllWindows()            

# Sobel combined
sobel = (sobel_imgy + sobel_imgx)
cv2.imshow("Sobel", sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()