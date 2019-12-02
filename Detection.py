# -*- coding: utf-8 -*-
"""
@author: cyx
Reference: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400
           https://github.com/matterport/Mask_RCNN
"""

import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import Prediction
from datetime import datetime
import pandas as pd
import csv
from threading import Timer

# Configuration of Mask r-cnn
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.6


# filter detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes): # only keep car/truck in detection results
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)

# record the number of parking
def record(number):
    Datetime = datetime.now().strftime('%Y/%m/%d %H:%M')
    if os.path.exists("./dataset/parked number.csv"):
        with open("./dataset/parked number.csv","a",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([Datetime, int(number)])
            print(Datetime)
    else:
        dataframe = pd.DataFrame({'Datetime':Datetime,'Occupancy':int(number)},index=[0])
        dataframe.to_csv("./dataset/parked number.csv", index = False, sep = ',')

# load trained weights file and pre-trained model
model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig())
model.load_weights("mask_rcnn_coco.h5", by_name=True)

video_capture = cv2.VideoCapture("parking_lot_2.mp4")

parked_car_boxes = None 
Frames = 0 # frame recorder
timer = 0 # time recorder
parked_number_old = 0 # the number of parking in previous frame
parked_number_new = 0 #the number of parking in frame now

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
            
    rgb_image = frame[:, :, ::-1] # Convert the image color to RGB
    results = model.detect([rgb_image], verbose=0) # use model to detect
    r = results[0] # only 1 frame, get the first result
        
    # First time detect, get the location of all cars
    if parked_car_boxes is None:
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        parked_number_new = parked_car_boxes.size/4
        timer = timer + 1
    else:
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        parked_number_old = parked_number_new # pass the number of previous frame to this frame
        for box in car_boxes:
            y1,x1,y2,x2 = box
            cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 255, 0), 1) # mark cars
                    
        # compute the overlaps between cars in this frame and previous frame
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
            # find the parking, if the car is parking, the overlap will be almost 1
            # if car is driving, overlap will be small.
            max_IoU_overlap = np.max(overlap_areas)
            if max_IoU_overlap < 0.85:
                parked_car_boxes = np.delete(parked_car_boxes,parking_area,axis=0)
        parked_number_new = parked_car_boxes.size/4 #the number of parking in this frame
        
        # to prevent detection errors, if in 5 consecutive frames, the parking number is the same, it will be the final result
        if parked_number_old == parked_number_new:
            Frames = Frames + 1
        else:
            Frames = 0
        if Frames != 0:
            if Frames%5 == 0:
                print(parked_number_new)
                if int(parked_number_new) >= 23: # if parking lot is full, predict the parking number in next time period
                    pred = Prediction.predict_future()
                    for i in pred: # select parking available time
                        if i < 170:
                            print(int(i))

        # record time, record parking number over a period of time
        timer = timer + 1
        if timer == 54000:
            record(parked_number_new)
            timer = 0
        
        parked_car_boxes = car_boxes
                 
        cv2.namedWindow('Parking',0)
        cv2.imshow('Parking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
