import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break