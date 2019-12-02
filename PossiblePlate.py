################################ Reference #######################################
#Open Source Code come from:
#https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python

import cv2
import numpy as np

###################################################################################################
class PossiblePlate:

    # constructor #################################################################################
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""
    # end constructor

# end class




