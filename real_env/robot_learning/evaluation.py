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
from learning import RealEnv 
################################################
weights_file = "weights/weights/data.pkl"
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=2000000, timeout=.1)
##################################################
RL_Agent_v1.load_model(weights_file)

def evaluate_v1(env, agent, max_episode_num = 100, max_steps = 500, save_file = "RL_agent",
             e_greedy = -1, e_max = 1, e_min = 0.5, e_decay_rate = 0.01, save_every = 5):

    # optimizer = optim.RMSprop(agent.parameters())
    for episode in range(max_episode_num):
        print("Starting Episode {} ...".format(episode))
        input("Put the disc on a initial position and press a key to continue...")
        env.reset()  		
        rewards = []
        state = env.state 
        for steps in range(max_steps):
            print("curr. step: {}/{}".format(steps,max_steps))
            action = agent.get_action(state, e_greedy = -1)
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

	evaluate_v1(env, agent, max_episode_num = 10, max_steps = 1000,
             e_greedy = 1, e_max = 1, e_min = 0.0005, e_decay_rate = 0.01, save_every = 5)


	RealEnv.webcam.release() 
	cv2.destroyAllWindows()

