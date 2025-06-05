import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

model = "res10_300x300_ssd_iter_140000.caffemodel"
weights = "deploy.prototxt"
mod = cv2.dnn.readNetFromCaffe(weights, model)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]  # height and width
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    mod.setInput(blob)
    det = mod.forward()

    for i in range(det.shape[2]): 
        confidence = det[0, 0, i, 2]
        if confidence > 0.5:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,f"confidence:{confidence*100:.1f}%",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,250),2)

    cv2.imshow("Face Detected ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
