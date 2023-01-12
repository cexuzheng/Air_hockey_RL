import cv2 
webcam = cv2.VideoCapture(2) 
while True: 
	ret,frame=webcam.read()
	if ret==True: 
		cv2.imshow("Webcam",frame)
		key=cv2.waitKey(1) # 1 ms
		if key==ord("q"):
			break 

webcam.release() 

cv2.destroyAllWindows()

