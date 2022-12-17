import numpy as np
from collections import namedtuple
import random
import pickle


Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward'))


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

class ReplayMemory(object):

    def __init__(self, capacity = 10000):
        self.memory = []
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            self.memory.pop( random.randint(0,self.capacity) )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def save(self, file = 'memory_pickle'):
        with open(file, 'wb') as fp:
            pickle.dump(self, fp)
    
    def save(self, file = 'memory_pickle'):
        with open(file, 'rb') as fp:
            self = pickle.load(fp)
    

import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.autograd import Variable
import random

class DQN(nn.Module):
    def __init__(self, n_actions = 17, input_size = 6):
        super(DQN, self).__init__()
        self.device = torch.device("cpu")
        self.linear1 = nn.Linear(input_size, 50, device=self.device)
        self.linear2 = nn.Linear(50, 50, device=self.device)
        self.linear3 = nn.Linear(50, n_actions, device=self.device)

    def forward(self, input):
        # the input is expected as (N, n_input)
        if not torch.is_tensor( input ):
            input = torch.tensor( input, dtype=torch.float32, device=self.device )

        if  isinstance(input, list) or isinstance(input, tuple):
            input = torch.stack( input )
        elif input.dim() == 1:
            input = torch.stack( [input] )
        # input.requires_grad_(False) 
        x = F.relu( self.linear1(input) ) 
        x = F.relu( self.linear2(  x  ) ) 
        x = self.linear3(x)
        return x 
      

class RL_Agent_v1(nn.Module):
    def __init__(self, n_actions = 17, input_size = 6, learning_rate=1e-3, batch_size = 128, gamma = 1-1e-2):
        super(RL_Agent_v1, self).__init__()
        self.device = torch.device("cpu")
        self.n_actions = n_actions
        self.gamma = gamma

        self.agent = DQN(n_actions, input_size).to(self.device)
        self.clipped = DQN(n_actions, input_size).to(self.device)
        self.clipped.load_state_dict( self.agent.state_dict() )
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        
        self.batch_size = batch_size
        self.memory = ReplayMemory()
    
    def get_action(self, state, e_greedy = -1):
        if not torch.is_tensor( state ):
            state = torch.tensor( state, dtype=torch.float32, device=self.device )
        # sample an action
        exp_exp_tradeoff = np.random.rand()
        if( e_greedy == -1 or exp_exp_tradeoff >= e_greedy):  # exploitation
            with torch.no_grad():
                Q = self.agent.forward(Variable(state))[0]
                action_sample = np.argmax( np.squeeze(Q.detach().cpu().numpy()))
        else:
            action_sample = np.random.choice( self.n_actions )
        return action_sample
    
    def save_model(self, save_file):
        torch.save(self.agent.state_dict(), save_file)

    def load_model(self, load_file):
        self.agent.load_state_dict( torch.load(load_file) )
        self.clipped.load_state_dict( self.agent.state_dict() )
    
    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))      # array of transitions -> transition of arrays
        states_batch = torch.tensor( np.array(batch.state), dtype=torch.float32, device=self.device, requires_grad=False )
        done_mask = torch.tensor(batch.done, requires_grad=False)
        state_action_values = self.agent.forward( states_batch ).gather(1, torch.tensor( [batch.action], device=self.device, requires_grad=False) )
        not_done_next_state = torch.tensor(  np.array([ batch.next_state[i] for i in range(self.batch_size) if not done_mask[i] ]), dtype=torch.float32, device=self.device, requires_grad=False )
    
        with torch.no_grad():
            next_state_values = torch.zeros( self.batch_size, device = self.device, requires_grad=False )
            next_state_values[ torch.logical_not( done_mask ) ] = self.clipped.forward( not_done_next_state ).max(1)[0].detach()
            expected_state_action_values = (next_state_values * self.gamma) + torch.tensor(batch.reward, device=self.device, requires_grad=False)
        
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.parameters():       # clip the range of the gradients
            param.grad.data.clamp_(-100, 100)
        self.optimizer.step()

    def step(self, state, action, done, next_state, reward):
        self.memory.push( state, action, done, next_state, reward )
        if( len(self.memory) >= self.batch_size):
            self.learn()

    
