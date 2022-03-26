import cv2 as cv
import numpy as np

cap = cv.VideoCapture("video.mp4")
ret, first_frame = cap.read()
first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
first_gray = cv.GaussianBlur(first_gray, (5, 5), 0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_frame = cv.GaussianBlur(gray_frame, (5, 5), 0)
    difference = cv.absdiff(first_gray, gray_frame)
    difference = cv.cvtColor(difference, cv.COLOR_GRAY2BGR)
    difference = cv.cvtColor(difference, cv.COLOR_RGB2HSV )
    
    for i in range(206):
        difference[np.all(difference == (0,0,50+i), axis=-1)] = (0,255,0)
    
    
    cv.imshow("difference", difference)
    cv.imshow("origin",frame)
    

    k = cv.waitKey(0)
    
    if k == ord("s"):
        new = np.hstack((frame,difference))
        cv.imwrite("hw0_109550027_2.png", new)
        break
cap.release()
cv.destroyAllWindows()