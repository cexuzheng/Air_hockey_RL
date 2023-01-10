import cv2 
import sys 
import copy 
import time 
import numpy as np

webcam = cv2.VideoCapture(3) 
disc_X, disc_Y = 0, 0 
while True: 
	ret,frame=webcam.read()
	if ret==True: 


		image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask_disc = cv2.inRange(image,(100,100,20), (120,255,250)) # blueish

		(countours_disc, _) = cv2.findcountours_disk(mask_disc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		if len(countours_disc) !=0:
			cnt = max(countours_disc, key = cv2.contourArea) # find the biggest contour 
			M=cv2.moments(cnt) # find the centres 
			
			cY = tuple(cnt[cnt[:, :, 1].argmin()][0])

			if M["m00"] != 0:
				cX = int(M["m10"]/M["m00"])
			else:
				cX = cX 

			# frame = cv2.cvtColor(mask_disk, cv2.COLOR_GRAY2BGR) // for dark background 
			cv2.circle(frame, (cX, cY[1]), 10, (0,255,0), -1)
			cv2.imshow("Webcam",frame)
		
		Key=cv2.waitKey(1) # 1 ms
		
	if Key==ord("q"):
		break 





	

webcam.release() 
cv2.destroyAllWindows()

