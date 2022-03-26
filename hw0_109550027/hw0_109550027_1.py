# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:11:21 2022

@author: chuch
"""


import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("image.png"))
if img is None:
    sys.exit("Could not read the image.")

cv.rectangle(img,(608,505),(721,616),(0,0,255),3)
cv.rectangle(img,(836,477),(916,557),(0,0,255),3)
cv.rectangle(img,(1073,600),(1180,726),(0,0,255),3)
cv.rectangle(img,(985,417),(1042,468),(0,0,255),3)
cv.rectangle(img,(994,346),(1042,383),(0,0,255),3)
cv.rectangle(img,(1042,282),(1088,314),(0,0,255),3)


cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("hw0_109550027_1.png", img)
