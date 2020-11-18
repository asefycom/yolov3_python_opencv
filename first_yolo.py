import cv2
import numpy as np

cap = cv2.VideoCapture(0)

coco_file = "coco.names"
coco_classes = []

with open(coco_file, "rt") as f:
    coco_classes = f.read().rstrip("\n").split("\n")

print(coco_classes)
print(len(coco_classes))

while True:
    success, frame = cap.read()

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break