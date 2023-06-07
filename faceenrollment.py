import os
import cv2
import time
import numpy as np
import glob
import torch
import json
import yaml
import sys

from utils.configs.metaInfo import *
from torch2trt import torch2trt
from torch2trt import TRTModule

from utils.pis_utils import *
from utils.general import ( non_max_suppression, scale_boxes, xyxy2xywh )
from utils.plots import Annotator, colors, save_one_box




detection_model = TRTModule()
detection_model.load_state_dict(torch.load(detect_model_trt_path))#.cuda()

recog_model = TRTModule()
recog_model.load_state_dict(torch.load(recog_model_trt_path))


# create Directory for saving enrolled embedding vectors
createDir(dirPath)


imgList = glob.glob("PISTestImages/*")[:1]#[:5]# + glob.glob("testYaw/*")[:1] 
print(imgList)
driverID = "sharif"
faceEmbedding = []
#video_path = "TestVideos/output_SHARIF.avi"
cap = cv2.VideoCapture(video_path)
while (True):
        _, img = cap.read()
    
        #cv2.imwrite("face_pose.png", frame)
        #img = cv2.imread(im)
        frame = img.copy()
        im0s = img.copy()
        img = cv2.resize(img, (640,640))
        #print("person")
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.ascontiguousarray(img)
        #with dt[0]:
        im = torch.from_numpy(im).cuda()#to(model.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        landDetTime = 0
        perDetTime = 0
        keyDetTime = 0
        recogTime = 0
        AGSTime = 0
        # Inference
        start = time.time()
        for i in range (0,50):
           pred = detection_model(im)
        perDetTime = time.time() - start
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        multiPerson = 0

        # Process predictions
        for i, det in enumerate(pred):  # per image
            imc = im0s.copy() #if save_crop else im0  # for save_crop
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = str(int(cls))  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    print(label)
                    if "Face" in label:
                    
                        #if int(xyxy[0].cpu().detach().numpy()) > int(frame.shape[1]/2):
                            img_crop = save_one_box(xyxy, imc, BGR=True)
                            recPred = recognize_face(img_crop, recog_model)#landmark_plot(personImg, landmark_model)
                            faceEmbedding.append(recPred.squeeze(0).detach().cpu())
                            print(len(faceEmbedding))
                    
                    if len(faceEmbedding) > sizeOfEmbedding:
                        StackedEmbTensor=torch.stack(faceEmbedding)
                        savePathPt = dirPath + driverID + ".pt"
                        print("Saving embedding for recognition. Name:", savePathPt, len(faceEmbedding))
                        torch.save(StackedEmbTensor, savePathPt)
                        
                        sys.exit()
                    #annotator = Annotator(frame, line_width=line_thickness, example=str(names))
                    #image = annotator.box_label(xyxy, label, color=colors(c, True))
                    #print(image.shape)
        #info = cv2.putText(landmamrkImg, 'FPS: ' + str(int(50/(perDetTime + landDetTime))), (35, 35), font, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
        #cv2.imshow("Detection with YoloV5s6", info)
        
        
        #if cv2.waitKey(100) & 0xFF == ord('q'):
            #print("##### Thank you for being with FS Solution #####")
            #time.sleep(5)
        #    break

        #print("FPS:", int(1/(perDetTime + landDetTime + keyDetTime + recogTime + AGSTime)))

        
