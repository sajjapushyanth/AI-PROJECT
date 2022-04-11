# Importing Libraries
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np


# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    # if hands are present in imageqq(frame)
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                # store height and width of image
                h,w,_ = img.shape
                # calculate and append cx, cy coordinates
                # of handmarks from image(frame) to lmList
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])

            mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

    if lmList != []:
        x1,y1 = lmList[4][1],lmList[4][2]  # store x,y coordinates of (tip of) thumb
        x2,y2 = lmList[8][1],lmList[8][2]  # store x,y coordinates of (tip of) index finger
        # draw circle on thumb and index finger tip
        cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
        cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)

        # draw line from tip of thumb to tip of index finger
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
        # calculate square root of the sum of
        # squares of the specified arguments.
        length = hypot(x2-x1,y2-y1)

        bright = np.interp(length,[15,220],[0,100])
        print(bright,length)
        sbc.set_brightness(int(bright))

        # Hand range 15 - 220
        # Brightness range 0 - 100

    # Display Video and when 'q' is entered, destroy
    # the window
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
