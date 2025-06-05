#vignette filter

def mask(img):

    h,w=img.shape[:2]
    x_knl=cv2.getGaussianKernel(w,w/4)
    y_knl=cv2.getGaussianKernel(h,h/4)
    kannel=y_knl*x_knl.T
    mask=kannel/kannel.max()

    img1=np.copy(img)
    for i in range(3):
        img1[:,:,i]=img1[:,:,i] * mask
    return img1

img=cv2.imread("ca .jpg")
vint=mask(img)
cv2.imshow("masked",vint)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Sobel edge detection

cap = cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    sobel_x=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=3)
    sobel_y=cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=3)
    sobel_comb=cv2.magnitude(sobel_x,sobel_y)
    sobel_combind=np.uint8(sobel_comb)
    cv2.imshow("original:",frame)
    cv2.imshow("edge:",sobel_combind)
    cv2.imshow("sobel_x",sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
