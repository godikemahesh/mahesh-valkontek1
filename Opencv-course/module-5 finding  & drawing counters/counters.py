import cv2
import matplotlib.pyplot as plt
import numpy as np

#finding counters and drawing
imagePath = 'shapes.jpg'
image = cv2.imread(imagePath)
# Convert to grayscale
imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
imageCopy1 = image.copy()
cv2.drawContours(imageCopy1, contours, -1, (0,0,255), 3)
plt.imshow(imageCopy1[:,:,::-1]);