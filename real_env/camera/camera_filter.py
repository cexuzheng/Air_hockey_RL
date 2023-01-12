import cv2 
import sys 
import copy 
import time 
import numpy as np

webcam = cv2.VideoCapture(0) 
cX = 0 
cY = 0
while True: 
	ret,frame=webcam.read()
	if ret==True: 
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(image,(100,100,20),(120,255,250))

		(contours, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) !=0:
			cnt = max(contours, key = cv2.contourArea) # find the biggest contour 
			M=cv2.moments(cnt) # find the centres 
			
			cY = tuple(cnt[cnt[:, :, 1].argmin()][0])

			if M["m00"] != 0:
				cX = int(M["m10"]/M["m00"])
			else:
				cX = cX 

			# frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) // for dark background 
			cv2.circle(frame, (cX, cY[1]), 10, (0,255,0), -1)
			cv2.imshow("Webcam",frame)
		
		Key=cv2.waitKey(1) # 1 ms
		
	if Key==ord("q"):
		break 

webcam.release() 
cv2.destroyAllWindows()

