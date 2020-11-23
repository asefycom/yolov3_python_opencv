import cv2
import numpy as np

cap = cv2.VideoCapture(0)

coco_file = "coco.names"
coco_classes = []
net_config = "cfg/yolov3.cfg"
net_weights = "cfg/yolov3.weights"

with open(coco_file, "rt") as f:
    coco_classes = f.read().rstrip("\n").split("\n")

print(coco_classes)
print(len(coco_classes))

net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, frame = cap.read()

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break