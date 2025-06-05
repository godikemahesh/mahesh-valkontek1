
import cv2
import numpy as np
model="mobilenet_iter_73000.caffemodel"
weights="deploy_mobilenet.prototxt"
classes="mobilenet_ssd_classes.txt"
with open(classes,"r") as file:
    data=file.read().split("\n")

net=cv2.dnn.readNetFromCaffe(weights,model)
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    h,w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,0.007843, (300, 300), 127.5)
    net.setInput(blob)
    det=net.forward()
    for i in range(det.shape[2]):
        conf=det[0,0,i,2]
        if conf>0.7:
            idx = int(det[0, 0, i, 1])
            box=det[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2=box.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            text=f"{data[idx]}:{conf*100:.1f}%"
            cv2.putText(frame,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("detected:",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
