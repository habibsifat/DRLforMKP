# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:47:54 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 00:18:35 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:08:00 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:39:28 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 07:05:07 2021

@author: User
"""



import test_env as env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from time import time
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
date = datetime.now().strftime('%m%d_%H_%M')
# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
max_train_ep = 300
max_test_ep = 400
max_episodes = sys.argv[1]
learning_rate = sys.argv[2]
N = sys.argv[3]
mul = sys.argv[4]
kc = sys.argv[5]
gamma = sys.argv[6]
reward_op = sys.argv[7]
state_op = sys.argv[8]
max_episodes = int(max_episodes)
learning_rate = float(learning_rate)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N = int(N)
mul = int(mul)
kc = int(kc)
gamma = float(gamma)
reward_op = int(reward_op)
state_op = int(state_op)
input_size = N*2 + 3 + kc
num_of_ac = N
class ActorCritic(nn.Module):
    def __init__(self,input_size, output_size, itemsize, knapsize):
        super(ActorCritic, self).__init__()
        embedsize = 512
        kernelsize = 2
        stride = 2
        cvnout = (embedsize - kernelsize)/stride + 1
        cvnout = int(cvnout)
        # if torch.cuda.is_available():
        #     self.Embedding = nn.Linear(input_size, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.conv = nn.Conv1d(1, 1, 1, 1).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)        
        #     self.hidden_1 = nn.Linear(embedsize*2, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.hidden_2 = nn.Linear(embedsize*2, embedsize*2).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.Embedding_pi = nn.Linear(embedsize*2, output_size).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        #     self.Embedding_v = nn.Linear(embedsize*2, 1).to(torch.device("cuda:0"), dtype=torch.float64, non_blocking=True)
        # else:
        self.Embedding = nn.Linear(input_size, embedsize, dtype=torch.float64)
        # self.conv = nn.Conv1d(1, 1, 1, 1, dtype=torch.float64)  
        self.conv = nn.Conv1d(1,1,kernelsize,stride,dtype=torch.float64 )
        self.hidden_1 = nn.Linear(cvnout, embedsize, dtype=torch.float64)
        # self.hidden_2 = nn.Linear(embedsize, embedsize, dtype=torch.float64)
        self.Embedding_pi = nn.Linear(embedsize, output_size, dtype=torch.float64)
        self.Embedding_v = nn.Linear(embedsize, 1, dtype=torch.float64)
        
        self.itemsize = itemsize
        self.knapsize = knapsize
        self._initialize_weights( -0.08,  0.08)
    def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

         
    def pi(self, x,  softmax_dim=1):
 
        
        data = x.clone().detach()

        # data = data.to(device)
        # data_conv = self.conv(data)
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))
        # embeded_out = torch.relu(self.hidden_2(embeded_out))
        u = torch.sigmoid(self.Embedding_pi(embeded_out).squeeze(1))
        # if torch.isnan(u[0,0]) : 
        #     print('nan')
        #     print(data)
            # print('embeded_out', embeded_out)
            # print(u)
            # self._initialize_weights( -0.08,  0.08)
    
            # embeded = self.Embedding(data)
            # embeded_out = torch.relu(self.hidden_1(embeded))
            # embeded_out = torch.relu(self.hidden_2(embeded_out))
            # u = torch.sigmoid(self.Embedding_pi(embeded_out).squeeze(0))
        # if max(u[0]) == 0:
        #     print('zero')
        #     print(data)
        #     u = u + 1e-40
        # prob = F.softmax(u, dim=softmax_dim)
        return u          
    
    def v(self, x, device, softmax_dim=1):
 
        
        data = x.clone().detach()

        data = data.to(device)

        
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))
        # embeded_out = torch.relu(self.hidden_2(embeded_out))
        u = self.Embedding_v(embeded_out).squeeze(1)
        # u = u + e6
        return u 


def test(global_model):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_of_ac = N

    rList = []
    sList = []
    idx = []
    last_state = []
    knap_map = []
    file = name = input("please write test_File name to open: (Source item)")
    # file = 'ep_1000_item_50_knap_3_R_10_R2_40_data.pickle_220108_16_36'
    envs = env.ENV(N,kc, reward_op, state_op, file)
    name = input("please write pt file name to open: ")
    global_model.load_state_dict(torch.load('./Pt/' + name))
 #   global_model.eval()
    t1 = time()

    for n_epi in range(max_episodes):
        done = False
        s = envs.reset(n_epi)
        


        while not done :
            s2 = np.expand_dims(s, 1)
            prob = global_model.pi(torch.from_numpy(s2).double())
            # prob = global_model.pi(s2)
            a = Categorical(prob).sample().numpy()
            s_prime, done = envs.step(a)

            s = s_prime
           

        last_state.append(envs.selected)

    t2 = time()
    computime = t2 - t1
    data = {
        # 'rList': rList,
        # 'sList': sList,
        'learning_rate': learning_rate,
        'compute_time':computime,
        # 'knap_map' : knap_map,
        'last_state': last_state
        # 'idx' : idx
    }         

    with open('test_%s_item_%d_knap_%d_data.pickle'%(date,  global_model.itemsize, global_model.knapsize), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    global_model = ActorCritic(input_size, num_of_ac, N, kc)

    test(global_model)
