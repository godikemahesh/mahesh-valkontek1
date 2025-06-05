import cv2
import matplotlib.pyplot as plt
import numpy as np

#converting Jpg to png
img = cv2.imread("mahathi sign.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrsh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

b, g, r = cv2.split(img)

print(r.shape, g.shape, b.shape, thrsh.shape)


png = cv2.merge((b, g, r, thrsh))

# Save the new image with alpha channel
cv2.imwrite("extract.png", png)
hist=cv2.calcHist([img],[0],None,[256],[0,255])
hist=cv2.calcHist([img],[1],None,[256],[0,255])
hist=cv2.calcHist([img],[2],None,[256],[0,255])
plt.subplot(111)
plt.plot(hist,"r")
plt.plot(hist,"g")
plt.plot(hist,"b")
plt.show()