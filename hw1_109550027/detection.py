import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime
import adaboost


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    file = open(dataPath,'r')
    lines = file.readlines()
    file.close()
    f = open("Adaboost_pred.txt", "w")
    cap = cv2.VideoCapture('data/detect/video.gif')
    first=1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("cant recieve frame anymore")
            break
        for i in range(1,int(lines[0])+1):
            line = lines[i].split(' ')
            xy=[0]*8
            for ind,s in enumerate(line):
                xy[ind]=int(s)
            image = crop(xy[0],xy[1],xy[2],xy[3],xy[4],xy[5],xy[6],xy[7],frame)
            image=cv2.resize(image,(36,16))
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result=clf.classify(image)
            
            if result==1:
                pts = np.array([[xy[0],xy[1]],[xy[2],xy[3]],[xy[6],xy[7]],[xy[4],xy[5]]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,(0,255,0),2)
                f.write('1 ')
            else:
                f.write('0 ')
        f.write('\n')
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        if(first==1):
            cv2.imwrite('frist_frame.png',frame)
        first=0
    
    f.close()
    cap.release()
    cv2.destroyAllWindows()    
    
    # End your code (Part 4)
