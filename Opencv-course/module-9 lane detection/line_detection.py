import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

#line detection
video=cv2.VideoCapture("videoplayback.mp4")
#video.set(cv2.CAP_PROP_FPS,10)
while True:
    ret,img=video.read()
    img1=img.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
    gray=cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)

    thresh=cv2.inRange(gray,200,255)
    roi_vertices=np.array([[[0,1910],
                  [1070,1910],
                  [1070,250],
                  [0,250]]])
    mask=np.zeros_like(thresh)
    cv2.fillPoly(mask,roi_vertices,255)
    roi=cv2.bitwise_and(thresh,mask)
    canny=cv2.Canny(roi,150,255)
    canny_blur=cv2.GaussianBlur(canny,(3,3),1)
    lines=cv2.HoughLinesP(canny_blur,1,np.pi/180,50,10,20)
    hogh=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.namedWindow("wind",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wind",300,600)
    cv2.imshow("wind",img1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
