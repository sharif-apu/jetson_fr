import os
import cv2
import time
import numpy as np
import glob
import torch
import json
import yaml

from torch2trt import torch2trt
from torch2trt import TRTModule

from utils.configs.metaInfo import *
from utils.pis_utils import *

from utils.general import ( non_max_suppression, scale_boxes, xyxy2xywh )
from utils.plots import Annotator, colors, save_one_box



detection_model = TRTModule()
detection_model.load_state_dict(torch.load(detect_model_trt_path))#.cuda()


recog_model = TRTModule()
recog_model.load_state_dict(torch.load(recog_model_trt_path))

fc = 0
recogFlag = 0
PAG = 0
DAG = 0
pasRec = 0
driRec = 0
avg_p = 0
fc_test = 100
driverAGS = []
passengerAGS = []
driverRecog = []
driverAGSEst = []
passengerAGSEst = []

faceEmbedding = []
pasReinit = 0 
drivReinit= 0 
driverID = None
cap = cv2.VideoCapture(video_path)

filename = 'video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, float(20),
                      (800, 1200), True)
#out = cv2.VideoWriter(output, fourcc, 20, (800,1400))
while (True):
    #for ic,im in enumerate(imgList):
    startTimeFull = time.time()
    ret, img = cap.read()
    if not ret:
        print("closing video")
        break 
    img = cv2.flip(img, 1)
    fc += 1
    frameOrg = img.copy()
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

    # Initiate Variable 
    driver_info = []
    passenger_info = []
    person_dict = {}
    landDetTime = 0
    perDetTime = 0
    keyDetTime = 0
    recogTime = 0
    AGSTime = 0
    

    # Inference
    start = time.time()
    pred = detection_model(im)
    perDetTime = time.time() - start
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    multiPerson = 0
    avg_fr = []
    avg_fr_wr = []
    # Process predictions
    try:
        for i, det in enumerate(pred):  # per image
            imc = im0s.copy() #if save_crop else im0  # for save_crop
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = str(int(cls))  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #print(label)
                    
                    if "Face" in label:
                        face_crop = save_one_box(xyxy, imc, BGR=True)
                        start = time.time()
                        
                        recPred = recognize_face(face_crop, recog_model) #landmark_plot(personImg, landmark_model)
                        recogTime += (time.time() - start)
                        faceEmbedding.append(recPred.squeeze(0).detach().cpu())
                        perID = faceEmbeddingMatching(torch.stack(faceEmbedding), dirPath, recog_threshold)
                                    
                    
                        annotator = Annotator(frame, line_width=line_thickness, example=str(names))
                        info = annotator.box_label(xyxy, perID, color=colors(c, True))
        #print(info)

        FPS = str(int(1/(time.time()-start)))

        vis = cv2.putText(info, 'FPS: ' + FPS, (dx, dy), font, .6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("PIS Demo", vis )
        
    except Exception as e:
        vis = frameOrg
        #print(vis.shape, e)
        cv2.imshow("PIS Demo", vis )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()



        
