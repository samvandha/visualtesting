import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
print("Camera Started")


def draw_polygon(contours, frame):
    largest_area = 0
    (xg_l, yg_l, wg_l, hg_l) = (0,0,0,0)
    k = 0
    flag = 0
    contours_that_satisfy= []
    if len(contours) != 0:
        for i in range(len(contours)):
            if cv.contourArea(contours[i]) > 90 and cv.contourArea (contours[i]) < 700:
                
                perimeter = cv.arcLength(contours[i], True)
                epsilon = 0.02* perimeter
                approx = cv.approxPolyDP(contours[i], epsilon, True)
                if len(approx) > 3:
                    (xg, yg, wg, hg) = cv.boundingRect(contours[i])
                    if (yg> 240) and yg < 400 and xg> 210 and xg < 430 and (xg+wg) < 430 and wg< 50:
                        
                        if wg* hg > largest_area:
                            largest_area = wg*hg
                            contour_with_largest_area = contours[i]
                            (xg_l, yg_l, wg_l, hg_l) = cv.boundingRect(contours[i])
                        else:
                            contours_that_satisfy.append(contours[i])
                            k += 1
                        #cmax = max(cmax, contours[i], key = cv.contourArea)
                        cv.rectangle(frame, (xg, yg), (xg+wg, yg+hg), (0, 255, 255), 2)
                        cv.rectangle(original, (xg, yg), (xg+wg, yg+hg), (0, 255, 255), 2)
                        cv.rectangle(original, (xg_l, yg_l), (xg_l+wg_l, yg_l+hg_l), (0, 0, 255), 2)
        if len(contours_that_satisfy) != 0:
            for j in range(len(contours_that_satisfy)):
                (xg, yg, wg, hg) = cv.boundingRect(contours_that_satisfy[j])
                if xg_l+10>xg and xg>xg_l-10 and yg_l+10>yg and yg-10<yg and hg < hg_l and wg_l+15>wg and wg_l-15<wg:
                    print("Washer Exists")
                    print("Hi")
                    flag = 1
               
    if flag == 0:
        print("Washer doesn't exist")
            
            

def show_horizontal_lines(image):
    horizontal = np.copy(image)
    cols = horizontal.shape[1]
    horizontal_size = cols //30
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    longest_horizontal = [horizontalStructure[0]]
##    for i in range(len(horizontalStructure)):
##        if horizontalStructure[i] > longest_horizontal:
##            longest_horizontal = horizontalStructure[i]
       
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
##    horizontal = cv.erode(horizontal, longest_horizontal)
##    horizontal = cv.erode(horizontal, longest_horizontal)
    cv.imshow("Horizontal", horizontal)


while cap.isOpened():
    _, frame = cap.read()
    original = frame.copy()
    frame1 = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    med = cv.medianBlur(gray, 3)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
    th4 = cv.adaptiveThreshold(med, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    th7 = th4.copy()
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    edged = cv.Canny(gray, 10, 30)
    contours_canny, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
    img_canny = cv.drawContours(frame1, contours_canny, -1, (255, 0,0), 1)
    kernel = np.ones((3,3), np.float32)/9
    dst = cv.filter2D(gray, -1, kernel)
    th5 = cv.adaptiveThreshold(edged, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 1)
    th6 = cv.adaptiveThreshold(dst,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 1)
    contours_box_blur, _ = cv.findContours(th6, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_median_adaptive, _ = cv.findContours(th4, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    draw_polygon(contours_box_blur, th6)
    #draw_polygon(contours_median_adaptive, th4)
    show_horizontal_lines(th7)
    cv.imshow("Original", original)
    cv.imshow("Gray", thresh)
    cv.imshow("BBox BLur", dst)
    cv.imshow("Median BLur", med)
    #cv.imshow("Canny", img_canny)
    #cv.imshow("Contours", img)
    cv.imshow("Gaussian", th2)
    cv.imshow("Mean", th3)
    cv.imshow("Median and Adaptive", th4)
    #cv.imshow("Canny and adaptive", th5)
    cv.imshow("Box Blur and Adaptive", th6)
    #cv.imshow("Original", frame)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
