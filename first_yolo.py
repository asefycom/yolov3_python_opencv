import cv2
import numpy as np

cap = cv2.VideoCapture(0)

coco_file = "coco.names"
coco_classes = []
net_config = "cfg/yolov3.cfg"
net_weights = "cfg/yolov3.weights"
blob_size = 320

with open(coco_file, "rt") as f:
    coco_classes = f.read().rstrip("\n").split("\n")

# print(coco_classes)
# print(len(coco_classes))

net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size,blob_size),mean=(0,0,0)
                                 ,swapRB=True,crop=False)
    # for image in blob:
    #     for k, b in enumerate(image):
    #         cv2.imshow(str(k), b)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    output = net.forward(out_names)
    # print(net.getUnconnectedOutLayersNames())
    # print(len(output))
    # print(type(output))
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)


    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break