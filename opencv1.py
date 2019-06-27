# In[1]:
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn import tree
import time




def nothing(x):
    pass

cv2.namedWindow("img")
cv2.createTrackbar("H-L", "img", 0, 179, nothing)
cv2.createTrackbar("H-U", "img", 179, 179, nothing)
cv2.createTrackbar("S-L", "img", 0, 255, nothing)
cv2.createTrackbar("S-U", "img", 255, 255, nothing)
cv2.createTrackbar("V-L", "img", 0, 255, nothing)
cv2.createTrackbar("V-U", "img", 255, 255, nothing)

cap = cv2.VideoCapture(1)



while True:

    H_L = cv2.getTrackbarPos("H-L", "img")
    H_U = cv2.getTrackbarPos("H-U", "img")
    S_L = cv2.getTrackbarPos("S-L", "img")
    S_U = cv2.getTrackbarPos("S-U", "img")
    V_L = cv2.getTrackbarPos("V-L", "img")
    V_U = cv2.getTrackbarPos("V-U", "img")

    key = cv2.waitKey(1)

    if key == 27:
        break

    history, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([H_L, S_L, V_L])
    upper_red = np.array([H_U, S_U, V_U])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

    cv2.imshow("img", frame)


cap.release()
cv2.destroyAllWindows()


