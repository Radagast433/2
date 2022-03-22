from torch import hub # Hub contains other models like FasterRCNN
import torch
import cv2
import imutils
import numpy as np
import time

# Encienda la cámara ip
video = "rtsp://admin:admin2018@192.168.14.123"  # http: // admin: admin @     Dirección /
#video = "rtsp://admin:admin@192.168.14.122"
cap = cv2.VideoCapture(video)
#capture = cv2.VideoCapture((0)

thres = 0.48 # Threshold to detect object
nms_threshold = 0.2
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(video)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    img = imutils.resize(img, width = 800, height = 600)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    classIds = list(classIds)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    #print(indices)
    indices = list(indices)
    #print(indices)
    
    for i in indices:
    
        try:
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            
            cv2.putText(img,classNames[int(classIds[i])-1].upper(),(int(box[0])+10,int(box[1])+30), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        
        except:
            pass
        
    cv2.imshow('Output',img)
   
    c= cv2.waitKey(10)
    if c == 27:
        # tecla clave para salir
        print("esc break...")
        break

cap.release()
cv2.destroyWindow("Output")





