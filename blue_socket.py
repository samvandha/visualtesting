import cv2 as cv
import os
import numpy as np
import time

##########################################################################################
# PATHAK-SINGH INTERNSHIP 2019 DECEMBER
# IF YOU GO THROUGH THE CODE YOU WILL SEE WE HAVE COMPARED THE NO OF CONTOURS FROM TWO FRAMES
# TO GET A IDEA OF THE PINS PRESENT IN THE BLUE SOCKET
# ONE BEING THE THRESHOLDING AS PINS ARE WHITE COLORED
# SECOND IS A FRAME RESULTING FROM THE MASKING OF WHITE ALONG WITH THE X COORDINATE OF THE PINS
# THE CONDITION IS ONLY SATISFIED IF THE PINS SATISFY THE DIMENSION PART AS WELL AS THE POSITION PART
# FURTHERMORE WHILE INTEGRATION, WE WILL BE TAKING AVERAGE OF THE VALUES GENERATED AND AVERAGING
##########################################################################################

cap = cv.VideoCapture(1)
print("Camera Started")
cap.set(cv.CAP_PROP_AUTOFOCUS, False)
cap.set(cv.CAP_PROP_FOCUS, 22)
os.chdir(r'C:\Users\matlab.pc\Desktop')

while cap.isOpened():
    cap.set(cv.CAP_PROP_FOCUS, 25)
    _, frame = cap.read()
    original = frame.copy()
    frame1 = frame.copy()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([90,102,33])
    upper = np.array([120,246,220])
    mask1 = cv.inRange(hsv, lower, upper)
    kernel = np.ones((1,1), np.uint8)
    mask = mask1
    blue_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    res_blue = cv.bitwise_and(frame, frame, mask=mask)
    med = cv.medianBlur(blue_mask, 3)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.dilate(gray, None, iterations=1)
    med = cv.erode(med, None, iterations=1)
    med_kernel = cv.erode(med, kernel, iterations= 1)
    contours_blue, _ = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #th4 = cv.adaptiveThreshold(blue_mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 3)
    if len(contours_blue) != 0:
        cmax = max(contours_blue, key = cv.contourArea)
        if cv.contourArea(cmax)> 300:
            (x, y, w, h) = cv.boundingRect(cmax)
            print("Width is: ", w)
            cv.rectangle(original, (x,y), (x+w, y+h), (0, 255, 0), 2)
            ROI = original[y:y+h,x:x+w]
            ROI_copy = ROI.copy()
            ROI_left = original[round(y+2*h/10):round(y+8*h/10),round(x+2*w/10):round(x+5*w/10)]
            ROI_left_copy = ROI_left.copy()
            ROI_left_copy_copy = ROI_left.copy()
            lower_white = np.array([0, 45, 0])
            upper_white = np.array([180, 255, 255])
            hsv_white = cv.cvtColor(ROI_left_copy, cv.COLOR_BGR2HSV)
            mask_white = cv.inRange(hsv_white, lower_white, upper_white)
            cv.imshow("Masking white", mask_white)
            gray_ROI = cv.cvtColor(ROI_left, cv.COLOR_BGR2GRAY)
            ret, thresh_ROI = cv.threshold(gray_ROI, 114,255, cv.THRESH_BINARY)
            #cv.imshow("Threshold from Gray", thresh_ROI)
            th4 = cv.adaptiveThreshold(gray_ROI, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)
            contours_th4, _ = cv.findContours(thresh_ROI, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            img = cv.drawContours(ROI_left_copy, contours_th4, -1 , (0, 20, 255), 1)
     
            if len(contours_th4) != 0:
                print("The number of contours detected are: ", len(contours_th4))
                for i in range(len(contours_th4)):
                    (xg, yg, wg, hg) = cv.boundingRect(contours_th4[i])
                    cv.rectangle(ROI_left_copy, (xg, yg), (xg+wg, yg+hg), (0,200,255), 1)
                 
            contours_mask_white_1, _ = cv.findContours(mask_white, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            print("No Contours from white mask", len(contours_mask_white_1))
            #print("First Contour from white mask",contours_mask_white_1[0])
            #time.sleep(10)
            
            contours_mask_white = []
            for cont in range(len(contours_mask_white_1)):
                useful_contour = 0
                print("Contour Area Value:",cv.contourArea(contours_mask_white_1[cont]))
                if cv.contourArea(contours_mask_white_1[cont]) > 35 and cv.contourArea(contours_mask_white_1[cont]) < 110:
                    (x1,y1,w1,h1) = cv.boundingRect(contours_mask_white_1[cont])
                    _, width = mask_white.shape
                    if x1>round(width/10) and x1< round(5*width/10) and w1< round(3*width/10):
                        contours_mask_white.append(contours_mask_white_1[cont])

            print("Contours from Useful white mask: ", len(contours_mask_white))
            #time.sleep(1)
            #contours_mask_white= contours_mask_white_1 
            if len(contours_th4) == len(contours_mask_white) == 2:
                print("Print the number of pins are: ", len(contours_th4))
            else:
                print("Error in number of pins")
            
                    #cv.imwrite("Pins.jpg", ROI_left_copy)
                    #cv.imwrite("BlueOriginal.jpg", original)
##            if len(contours_th4) != 0:
##                for i in range(len(contours_th4)):
##                    if cv.contourArea(contours_th4[i]) < 1000 and cv.contourArea(contours_th4[i]) > 20 :
##                        print("Contours matching found")
##                        perimeter = cv.arcLength(contours_th4[i], True)
##                        epsilon = 0.1*perimeter
##                        approx = cv.approxPolyDP(contours_th4[i], epsilon, True)
##                        if len(approx) > 4:
##                            print("Matching Approx found")
##                            (xg, yg, wg, hg) = cv.boundingRect(contours_th4[i])
##                            cv.rectangle(ROI_left, (xg,yg), (xg+wg,yg+hg), (0,0,200), 1)  
            cv.imshow("ROI", ROI)
            cv.imshow("ROI LEFT COPY", ROI_left_copy)
            cv.imshow("ROI LEFT COPY COPY", ROI_left_copy_copy)
            #cv.imshow("ROI LEFT", ROI_left)
            #cv.imshow("Contours on ROI", img)
            cv.imshow("Threshold from Blue Mask", th4)
            cv.imshow("Thresh  ROI", thresh_ROI)
            
    #cv.imshow("Blue", blue_mask)
    #cv.imshow("Blue Blur", med)
    #cv.imshow("Blue Blur with kernel", med_kernel)
    cv.imshow("Original", original)
    #cv.imshow("Res Mask", res_blue)

    #cv.imshow("Contours", img)
    k = cv.waitKey(100) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
