#! /usr/bin/env python
import sys
import copy
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import skimage
from skimage import measure


image = cv2.imread('20221012_180211.jpg', cv2.IMREAD_COLOR)
image_original = image


current_time=time.time()

#Red  color 
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 120, 60], dtype="uint8")
upper = np.array([7, 255, 240], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
lower = np.array([171, 120, 60], dtype="uint8")
upper = np.array([180, 255, 240], dtype="uint8")
mask2 = cv2.inRange(image, lower, upper)
mask= mask+mask2

(contours, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key = cv2.contourArea)

M=cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print(time.time()-current_time)


print(cX, cY)
image_final = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.circle(image_final, (cX, cY), 5, (0,0,255), -1)
cv2.imshow("result", image_final)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

