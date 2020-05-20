##########################################################################################
# PATHAK-SINGH INTERNSHIP 2019 DECEMBER
##########################################################################################
import cv2 as cv
import numpy as np
import os

directory = r'C:\Users\matlab.pc\Desktop'
os.chdir(directory)
cap = cv.VideoCapture(0)
print("Camera has been started")
cap.set(cv.CAP_PROP_AUTOFOCUS, False)
cap.set(cv.CAP_PROP_FOCUS, 22)
circles = []
red_circles = []
nosal_hole_radius = 15
largest_radius = 0
circle_found = 0
radius_circles = []

def average(lst):
    return sum(lst)/len(lst)
def rotate_image(image):
    image_center = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv.getRotationMatrix2D(image_center, 180, 1)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv.INTER_LINEAR)
    return result
while cap.isOpened():
    cap.set(cv.CAP_PROP_FOCUS, 25)
    ret, frame = cap.read()
    original = frame.copy()
  
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 200)
    ret, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
    th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue1 = np.array([90,102,33])
    upper_blue1 = np.array([120,246,220])
    mask = cv.inRange(hsv, lower_blue1, upper_blue1)
    cv.imshow("Gray_frame", mask)
    blue_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    res_blue = cv.bitwise_and(frame, frame, mask = blue_mask)

    med = cv.medianBlur(gray, 3)
    th5 = cv.adaptiveThreshold(med, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    #th4 = cv.adaptiveThreshold(gray_black, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 3)
    contours_blue, _ = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours_blue)!= 0:
        cmax = max(contours_blue, key = cv.contourArea)
        if cv.contourArea(cmax) > 500:
            (xg, yg, wg, hg) = cv.boundingRect(cmax)
            ROI_blue_for_logo = original[round(yg+hg/3):round(yg+2*hg/3), round(xg+2*wg/5):round (xg+9*wg/10)]
            rotated = rotate_image(ROI_blue_for_logo)
            rotated_gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
            med1= cv.medianBlur(rotated_gray, 3)
            th6 = cv.adaptiveThreshold(med1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7,2)
            cv.imshow("Rotated", rotated)
            cv.imshow("Rotated Gray", th6)
            xg = xg + round(6*wg/5)
            yg = yg + round(hg/3)
            wg = round(wg/5)
            hg = round(hg/3)
            cv.rectangle(original, (xg, yg), (xg+wg, yg+hg), (255, 0, 255), 2)
            ROI_adaptive = th5[yg:yg+hg, xg:xg+wg]
            ROI_COPY = frame[yg:yg+hg, xg:xg+wg]
            ROI = frame[yg:yg+hg, xg:xg+wg]

            circles = cv.HoughCircles(ROI_adaptive, cv.HOUGH_GRADIENT, 1, 20, param1=5,param2=15,minRadius=0,maxRadius=9)
            if circles is not None:
                print("Circle detected", len(circles))
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) == 1:
                    for (x,y,r) in circles:
                        cv.circle(ROI_COPY, (x,y), r, (0, 255, 0), 2)
                        print("The radius of the circle is: ", r)
                        #cv.imwrite('RedHole.jpg', ROI_COPY)
                        radius_circles.append(r)
                        circle_found = 1
                    
            cv.imshow("ROI_COPY", ROI_COPY)
            cv.imshow("ROI_ADAPTIVE", ROI_adaptive)
            cv.imshow("ROI", ROI)
    cv.imshow("Original", original)
    cv.imshow("Median and adaptive", th5)
    cv.imshow("Frame", frame)
    
    k = cv.waitKey(1) & 0xFF
    if k ==27:
        break

cv.destroyAllWindows()
cap.release()
#cv.imwrite('Object.jpg', original)
if circle_found == 1:
    print("We have found the red nosal hole", average(radius_circles))   
else:
    print("Didn't find the nosal hole")
