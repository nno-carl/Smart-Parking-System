################################ Reference #######################################
#Open Source Code come from:
#https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python

import cv2
import os
import datetime 

import DetectChars
import DetectPlates
import Mongo

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main(filename):

    testKNN = DetectChars.loadKNNDataAndTrainKNN()         

    if testKNN == False:                               
        print("\nerror: KNN traning was not successful\n")  
        return                                                          

    imgOriginalScene  = cv2.imread(filename)               

    if imgOriginalScene is None:                            
        print("\nerror: image not read from file \n\n")  
        os.system("pause")                                  
        return

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)    #Find possible license plates(not match yet)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)    # the license plate which get recognized of specific content after processing


    if len(listOfPossiblePlates) == 0:                          
        print("\nno license plates were detected\n") 
    else:                                                       

                # sort recognized plates in DESCENDING order based on the number of recognized chars
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # the one with the most recognized chars is real plate
        licPlate = listOfPossiblePlates[0]


        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n\n")  
            return                                          
    


        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  
        print("----------------------------------------")
        
        localtime = datetime.datetime.now()
        date = localtime.strftime('%Y.%m.%d')
        localtime = localtime.strftime('%Y-%m-%d-%H:%M:%S')
        enterTime, leaveTime, duration = Mongo.insert(licPlate.strChars,date,localtime,localtime)


    return licPlate.strChars,enterTime,leaveTime,duration



if __name__ == "__main__":
    main()


















