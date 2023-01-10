###############################################
import cv2 
import sys 
import copy 
import time 
import numpy as np
from collections import namedtuple
import serial 
import random
import pickle 
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.autograd import Variable

################################################
from ..game_engyne import air_hockey
from ..agent import DQN, ReplayMemory, RL_Agent_v1
################################################
arduino = serial.Serial(port='/dev/ttyUSB1', baudrate=2000000, timeout=.1)


def write_list(memory, file = 'memory_pickle'):
    # store list in binary file so 'wb' mode
    with open(file, 'wb') as fp:
        pickle.dump(memory, fp)

# Read list to memory
def read_list(file = 'memory_pickle'):
    # for reading also binary mode is important
    with open(file, 'rb') as fp:
        memory = pickle.load(fp)
        return memory


# ENV: 
class real_env(): 
	def __init__(self, POS1_img_plane = (316,220), POS2_img_plane=(765,217), pixel_thrshold = 20):
		self.discX_state, self.discY_state = self.read_arduino()
		self.POS1=POS1_img_plane     # x,y in image plane
		self.POS2=POS2_img_plane
		self.pixel_thrshold = pixel_thrshold 
	
	def reset(self): 
		self.x_state = 0.20 # Initial state in cartesian coordinates.
		self.y_state = 0
		self.write_arduino() # send initial coordinates without any increments
		self.

	## CAMERA (DISC): 
	def get_disc_state(self,frame): 
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask_disc = cv2.inRange(image,(100,100,20), (120,255,250)) # blueish
		(countours_disc, _) = cv2.findcountours_disk(mask_disc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		if len(countours_disc) !=0:
			cnt = max(countours_disc, key = cv2.contourArea) # find the biggest contour 
			M = cv2.moments(cnt) # find the centres 
			
			self.discY_state = tuple(cnt[cnt[:, :, 1].argmin()][0])

			if M["m00"] != 0:
				self.discX_state = int(M["m10"]/M["m00"])
				discX_state = self.discX_state 
			else:
				self.discX_state = discX_state 

	## ARDUINO (ROBOT): 
	def read_arduino(self):
		data = arduino.readline()
	 	#TODO----------------------------------------------
		
	def write_arduino(self, incr_x = 0.0, incr_y = 0.0):
		state_str = "{\"x\":%f, \"y\":%f }".format(self.discX_state+incr_x, self.discY_state+incr_y)
		arduino.write(bytes(state_str, 'utf-8'))

##################################################
webcam = cv2.VideoCapture(3) 
discX, discY = 0, 0 
while True: 
	ret,frame=webcam.read()
	if ret: 
		

			# frame = cv2.cvtColor(mask_disk, cv2.COLOR_GRAY2BGR) // for dark background 
			cv2.circle(frame, (discX, discY[1]), 10, (0,255,0), -1)
			cv2.imshow("Webcam",frame)
		
		Key=cv2.waitKey(1) # 1 ms
		
	if Key==ord("q"):
		break 


if __name__ == '__main__': 
	webcam = cv2.VideoCapture(3) 



	

webcam.release() 
cv2.destroyAllWindows()

