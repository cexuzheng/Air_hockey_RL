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
import json 

################################################
from agent import RL_Agent_v1
################################################

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=2000000, timeout=.1)

######################################################################################
# POS1_img_plane = (318,167), POS2_img_plane=(760,184) % Calibrated using camera images
########################################################################################

# ENV: 
class RealEnv(): 
	def __init__(self, pixel_thrshold = 20, action_increment = 0.005, camera_num = 0, POS1_img_plane = (0,0), POS2_img_plane=(0,0)):
		self.pixel_thrshold = pixel_thrshold 
		self.action_increment = action_increment
		self.action_space = np.array(((0, 0),(0, 1),(1, 0),(-1, 0),(0, -1)))
		self.webcam = cv2.VideoCapture(camera_num) 
		self.POS1 = POS1_img_plane
		self.POS2 = POS2_img_plane
		self.discX_state = 0
		self.discY_state = 0
		self.discState = 0 
		self.INITIAL_STATE= (500,220) 
		                            
	def reset(self): 
		self.robotX_state = 0.20 # Initial state in cartesian coordinates.
		self.robotY_state = 0
		self.write_arduino() # send initial coordinates without any increments
		self.read_arduino()		
		state = False 
		while state == False: # ensure get_disc_state properly returns the disk coordinates.  
			state = self.get_disc_state() # self.discX_state,self.discY_state
		
		if self.discState == 1:
			self.INITIAL_STATE = self.POS1
		if self.discState == 2:
			self.INITIAL_STATE = self.POS2
		print("Episode objective: POS {}".format(self.discState))
		self.state = (self.discState, self.robotX_state,self.robotY_state)

	def step(self, action):
		reward = 0; done = False; info = ()
		# actuate 
		self.write_arduino(self.action_increment*self.action_space[action][0], self.action_increment*self.action_space[action][1])
		time.sleep(0.1) # wait the specified s
		# get the new states: 	
		state = False 
		while state == False: # ensure get_disc_state properly returns the disk coordinates.  
			if self.discState == 3: 
				done = True
				break
			state = self.get_disc_state() # self.discX_state,self.discY_state

		self.read_arduino() # self.robotX_state,self.robotY_state

		new_state = (self.discState,self.robotX_state,self.robotY_state)
		
		# check if done  (disk moved from it's initial position) and give rewards
		a = np.array((self.discX_state, self.discY_state))
		b = self.INITIAL_STATE
		moved_dist = np.linalg.norm(a-b)
		if (moved_dist > self.pixel_thrshold):
			reward = 100 
			done = True 
		return (new_state, reward, done, info)

	## CAMERA (DISC): 
	def get_disc_state(self):
		succes = False  
		while succes == False:
			ret, frame = self.webcam.read()
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			mask_disc = cv2.inRange(image,(100,100,20), (120,255,250)) # blueish
			(contours_disc, _) = cv2.findContours(mask_disc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours_disc) !=0:
				cnt = max(contours_disc, key = cv2.contourArea) # find the biggest contour 
				M = cv2.moments(cnt) # find the centres 
				cY = tuple(cnt[cnt[:, :, 1].argmin()][0])
				
				self.discY_state = cY[1]

				if M["m00"] != 0:
					self.discX_state = int(M["m10"]/M["m00"]) 
				
				disc_state = np.array((self.discX_state, self.discY_state))

				dist_to_POS1 = np.linalg.norm(disc_state-self.POS1)
				dist_to_POS2 = np.linalg.norm(disc_state-self.POS2)

				if (dist_to_POS1 < self.pixel_thrshold):
					self.discState = 1
					succes = True 
				elif(dist_to_POS2 < self.pixel_thrshold):
					self.discState = 2
					succes = True 
				else:
					self.discState = 3
					succes = True
		return succes 

	## ARDUINO (ROBOT):  
	def read_arduino(self):
		value = arduino.readline().decode()
		json_is_valid=True
		try:
			data = json.loads(value)
		except ValueError as e:
			json_is_valid = False 		
		if(json_is_valid):
			self.robotX_state = data["x"] 
			self.robotY_state = data["y"] 
			

	def write_arduino(self, incr_x = 0.0, incr_y = 0.0):
		state_str = '{\"x\":%f, \"y\":%f} }'%(self.robotX_state+incr_x, self.robotY_state+incr_y)
		arduino.write(bytes(state_str, 'utf-8'))

##################################################

def train_v1(env, agent, max_episode_num = 100, max_steps = 500, save_file = "RL_agent",
             e_greedy = -1, e_max = 1, e_min = 0.5, e_decay_rate = 0.01, save_every = 5):

    # optimizer = optim.RMSprop(agent.parameters())
    for episode in range(max_episode_num):
        print("Starting Episode {} ...".format(episode))
        input("Put the disc on a initial position and press a key to continue...")
        env.reset()  		
        rewards = []
        state = env.state 
        if(e_greedy != -1):
            e_greedy = e_min + (e_max - e_min)*np.exp(-e_decay_rate*episode)

        for steps in range(max_steps):
            print("curr. step: {}/{}".format(steps,max_steps))
            action = agent.get_action(state, e_greedy)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            agent.step( state, action, done, new_state, reward )
            if done:
                print("Episode done!")
                break
            state = new_state

        if ( episode+1 ) % save_every == 0:        # update the clipped
            agent.save_model(save_file+'_'+str(episode))


if __name__ == '__main__': 
	env = RealEnv(pixel_thrshold = 50, action_increment = 0.03, camera_num = 0, POS1_img_plane = (318,187), POS2_img_plane=(760,184))
	agent = RL_Agent_v1(n_actions = 5, input_size = 3, learning_rate=1e-4, batch_size = 4, tau = 1e-4, learn_mode = 'clipped',
                gamma = 1-1e-1, nlayer1 = 32, nlayer2 = 32, memory_capacity = 1000, dtype = torch.float32 ) 
	train_v1(env, agent, max_episode_num = 100, max_steps = 1000, save_file = "RL_agent",
             e_greedy = 1, e_max = 1, e_min = 0.0005, e_decay_rate = 0.01, save_every = 5)
	RealEnv.webcam.release() 
	cv2.destroyAllWindows()

