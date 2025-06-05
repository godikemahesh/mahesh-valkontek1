dictionary=aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
#marker1=aruco.generateImageMarker(dictionary,25,200)
marker1=cv2.imread("markerImg1.png",cv2.IMREAD_GRAYSCALE)

corner,id,rej=aruco.detectMarkers(marker1,dictionary)
print(f"{id},{corner}")

aruco.drawDetectedMarkers(marker1,corner,id)
plt.imshow(marker1)
plt.show()
