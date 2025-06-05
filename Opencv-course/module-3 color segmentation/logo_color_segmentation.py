
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Mask specific color using inRange function

img=cv2.imread("opencv_logo.webp",cv2.IMREAD_COLOR)
nimg=img[30:400,0:500]
img_hsv=cv2.cvtColor(nimg,cv2.COLOR_BGR2HSV)
rlb=np.array([165,50,50],np.uint8)
rup=np.array([180,255,255],np.uint8)
g_lb = np.array([35, 50, 50], np.uint8)
g_ub = np.array([80, 255, 255], np.uint8)

# Set range for blue color.
b_lb = np.array([95, 50, 50], np.uint8)
b_ub = np.array([125, 255, 255], np.uint8)

g_mask = cv2.inRange(img_hsv, g_lb, g_ub)
b_mask = cv2.inRange(img_hsv, b_lb, b_ub)
rmask=cv2.inRange(img_hsv,rlb,rup)
plt.figure(figsize = (18, 4))


plt.subplot(131)
r_sg=cv2.bitwise_and(nimg,nimg,mask=rmask)
plt.imshow(r_sg[:,:,::-1])
plt.subplot(133)
b_sg=cv2.bitwise_and(nimg,nimg,mask=b_mask)
plt.imshow(b_sg[:,:,::-1])
plt.subplot(132)
g_sg=cv2.bitwise_and(nimg,nimg,mask=g_mask)
plt.imshow(g_sg[:,:,::-1])
plt.show()

