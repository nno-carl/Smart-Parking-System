import sys
import os
import cv2
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import Prediction
from datetime import datetime
import pandas as pd
import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from mainForm import Ui_MainWindow
import Main
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

    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]: # only keep car/truck in detection results
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

class PyQtMainEntry(QMainWindow, Ui_MainWindow):


    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera = cv2.VideoCapture("parking_lot_2.mp4")
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)
        self._timer.start() # loop each frame
    # read entrance plate image
    def btnReadImage_en_Clicked(self):
        self._timer.stop()
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open pic')
        if filename:
            self.captured = cv2.imread(str(filename))
            self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)
            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            plate,enterTime,leaveTime,duration = Main.main(str(filename))
            self.labelResult.setText(plate + "\n" + enterTime)
            print(plate)
            print(enterTime)
    # read exit plate image
    def btnReadImage_ex_Clicked(self):
        self._timer.stop()
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open pic')
        if filename:
            self.captured = cv2.imread(str(filename))
            self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)
            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            plate,enterTime,leaveTime,duration = Main.main(str(filename))
            self.labelResult.setText(plate+ "\n" + leaveTime + "\n" + str(duration))
            print(plate)
            print(leaveTime)
            print(str(duration))

    # process each frame
    @QtCore.pyqtSlot()
    def _queryFrame(self):
        global parked_car_boxes
        global Frames
        global timer
        global parked_number_old
        global parked_number_new
        global model
        ret, self.frame = self.camera.read()
        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        rgb_image = self.frame[:, :, ::-1] # Convert the image color to RGB
        results = model.detect([rgb_image], verbose=0) # use model to detect
        r = results[0] # only 1 frame, get the first result

        # First time detect, get the location of all cars
        if parked_car_boxes is None:
            parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            parked_number_new = parked_car_boxes.size/4
            parked_num = int(parked_number_new)
            self.labelResult.setText("Parking Number\n%d/30" % parked_num)
            timer = timer + 1
        else:
            car_boxes = get_car_boxes(r['rois'], r['class_ids']) # cars in this frame
            parked_number_old = parked_number_new # pass the number of previous frame to this frame
            for box in car_boxes:
                y1,x1,y2,x2 = box
                cv2.rectangle(self.frame,(x1, y1), (x2, y2), (0, 255, 0), 1) # mark cars

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
                    parked_num = int(parked_number_new)
                    print(parked_num)
                    self.labelResult.setText("Parking Number\n%d/30" % parked_num)
                    if int(parked_number_new) >= 20: # if parking lot is full, predict the parking number in next time period
                        pred = Prediction.predict_future()
                        j = 1
                        for i in pred: # select parking available time
                            if i < 160:
                                time = j * 30
                                self.Reminder(time)
                                print(int(i))
                                break
                            else:
                                j = j + 1
            # record time, record parking number over a period of time
            timer = timer + 1
            if timer == 54000:
                record(parked_number_new)
                timer = 0
            parked_car_boxes = car_boxes
        # show each frame of video    
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def Reminder(self,time):
        QMessageBox.information( self, "Reminder", "Parking is full, %d minutes will have free slots" % time )


if __name__ == "__main__":
    model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig()) # load trained weights file and pre-trained model
    model.load_weights("mask_rcnn_coco.h5", by_name=True)
    parked_car_boxes = None
    Frames = 0 # frame recorder
    timer = 0 # time recorder
    parked_number_old = 0 # the number of parking in previous frame
    parked_number_new = 0 #the number of parking in frame now
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())