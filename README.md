# Smart-Parking-System
It is a project for course Ubiquitous Sensing for Smart Cities.
We complete a smart parking system with 3 features:
Automatic license plate recognition.
Monitor parking numbers in real time.
Predict available parking time when parking lot is full.

tools as follow:
Python
Numpy
Pandas
Open-CV
Mask R-CNN
Scikit-learn
PyQt5
MongoDB

Uploaded files do not contain Mask R-CNN model.
Please go to https://github.com/matterport/Mask_RCNN/releases to download "mask_rcnn_coco.h5".

Before opening it, please make sure the above tools have been installed successfully.
Please install Mask R-CNN before open it. Download Mask R-CNN: https://github.com/matterport/Mask_RCNN
Do not forget to install MongoDB.

Run 'mainEntry.py' to open the project. It will automatically open video and detect cars.
The video and the number of parking will be shown on user interface.
If the number of parking over a threshold, it will predict available parking time.
The available parking time will be shown in a reminder.
Click button to input image of plate to recognize the number of the plate.
Enter and leave time，plate number and duration will be shown.


Reference:
https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400
https://github.com/matterport/Mask_RCNN
https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python
