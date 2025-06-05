import cv2
import numpy as np
import mediapipe as mp
import pyautogui as pg


mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mpdraw = mp.solutions.drawing_utils
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
wsrn, hscrn = pg.size()


plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 7

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgrgb)

    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks:
            imlist = []
            for id, lm in enumerate(handlms.landmark):
                cx, cy = int(lm.x * wcam), int(lm.y * hcam)
                imlist.append((cx, cy))

            if imlist:
                x1, y1 = imlist[8]
                x2, y2 = imlist[12]

                # Convert to screen coordinates
                scrn_x = np.interp(x1, (0, wcam), (0, wsrn))
                scrn_y = np.interp(y1, (0, hcam), (0, hscrn))

                # Smooth movement
                clocX = plocX + (scrn_x - plocX) / smoothening
                clocY = plocY + (scrn_y - plocY) / smoothening

                pg.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY


                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if distance < 30:
                    pg.click()

            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    cv2.imshow("Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
