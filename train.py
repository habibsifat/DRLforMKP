# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:07:53 2022

@author: gwsur
"""


import train_env as env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from datetime import datetime
import sys
import numpy as np
import pickle
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
load_op = sys.argv[9]
max_episodes = int(max_episodes)
learning_rate = float(learning_rate)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N = int(N)
mul = int(mul)
kc = int(kc)
gamma = float(gamma)
reward_op = int(reward_op)
state_op = int(state_op)
load_op = int(load_op)
input_size = N*2 + 3 + kc
num_of_ac = N
# try:
#     train_file = input("please write train_file name to open: (source item)")
# except EOFError:
#     print ('Why did you do an EOF on me?')
# try:
#     test_file = input("please write test_file name to open: (source item)") 
# except EOFError:
#     print ('Why did you do an EOF on me?')
# train_file = name = input("please write train_File name to open: (Source item)")
train_file = 'ep_1000_item_50_knap_3_R_10_R2_80_data.pickle_221103_12_34'

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
# class ActorCritic(nn.Module):
#     def __init__(self):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(4, 256)
#         self.fc_pi = nn.Linear(256, 2)
#         self.fc_v = nn.Linear(256, 1)

#     def pi(self, x, softmax_dim=0):
#         x = F.relu(self.fc1(x))
#         x = self.fc_pi(x)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob

#     def v(self, x):
#         x = F.relu(self.fc1(x))
#         v = self.fc_v(x)
#         return v
         
    def pi(self, x, device, softmax_dim=1):
 
        
        data = x.clone().detach()

        data = data.to(device)
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
    def pi_cpu(self, x, softmax_dim=1):
 
        
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
    
    def v_cpu(self, x,  softmax_dim=1):
 
        
        data = x.clone().detach()

      #  data = data.to(device)

        
        embeded = self.Embedding(data)
        conved = self.conv(embeded)
        embeded_out = torch.relu(self.hidden_1(conved))
        # embeded_out = torch.relu(self.hidden_2(embeded_out))
        u = self.Embedding_v(embeded_out).squeeze(1)
        # u = u + e6
        return u 

def train(global_model, rank):

    date = datetime.now().strftime('%m%d_%H_%M')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_of_ac = N
 
    rList = []
    sList = []
    lList = []
    envs = env.ENV(N,kc, reward_op, state_op, train_file)
    local_model = ActorCritic(input_size, num_of_ac, N, kc)
    local_model.load_state_dict(global_model.state_dict())
    # local_model = local_model.to(device)
    optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)
    each_peri_value_list = []
    total_value_list = []
    for n_epi in range(max_episodes*mul):
        
        total_reward = 0.0;  
        total_reward_list = []
        # selected_list = []
        total_loss = []
        done = False
        p = n_epi%max_episodes
        s = envs.reset(p)
        step_count  = 0 
        s_lst, a_lst, r_lst = [], [], []
        while not done:
            
            for j in range(update_interval):
                s2 = np.expand_dims(s, 1)
                prob = local_model.pi_cpu(torch.from_numpy(s2).double())
                a = Categorical(prob).sample().numpy()
                s_prime, r, done = envs.step(a)
                # if r > 0:
                #     r = r/100
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                total_reward +=r
                total_reward_list.append(r)
                # selected_list.append(envs.selected*envs.item_val)
                s = s_prime
                step_count +=1
                if done:
                    break
                    
                # s_final = torch.tensor(s_prime, dtype=torch.float)
            s_final = np.expand_dims(s_prime, 1)
            R = 0.0 if done else local_model.v_cpu(torch.from_numpy(s_final).double()).detach().clone().numpy()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()
            td_target = torch.tensor(td_target_lst)
            td_target = td_target.reshape(-1)            
            # s_vec = torch.tensor(s_lst).reshape(-1, input_size).double()  # input_size == Dimension of state
            s_vec = torch.tensor(s_lst).reshape(-1, input_size).double()  # input_size == Dimension of state
            # a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1).to(device)
            a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
            # td_target = td_target.to(device)
            # local_model = local_model.to(device)    
            # advantage = td_target - local_model.v(s_batch)
            advantage = td_target - local_model.v_cpu(s_vec.unsqueeze(1)).reshape(-1)
            pi = local_model.pi_cpu(s_vec.unsqueeze(1),  softmax_dim=1)
            pi_a = pi.gather(1, a_vec).reshape(-1)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.mse_loss(local_model.v_cpu(s_vec.unsqueeze(1)).reshape(-1), td_target.double())
            total_loss.append(sum(loss.detach().numpy().reshape(-1)))
            optimizer.zero_grad()
            loss.mean().backward()
            # local_model = local_model.to("cpu")
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
            # sum_val = max(np.sum(np.array(selected_list), axis=1))
            # each_peri_value_list.append(sum_val)
           # local_model = local_model.to(device)
            s_lst, a_lst, r_lst = [], [], []
        lList.append(sum(total_loss))
        rList.append(sum(total_reward))
        sList.append(step_count)   
        # if (n_epi+1)%(max_episodes*10) == 0 :
        #     print('rank %d, %d'%(int(rank), sum(each_peri_value_list)))
        #     total_value_list.append(sum(each_peri_value_list))
        #     each_peri_value_list = []
        #     torch.save(global_model.state_dict(),  './Pt/a3c_knap_real_middle_%s_item_%d_knap_%d_epi_%d_rank_%d_act.pt'%(date,  global_model.itemsize, global_model.knapsize, n_epi, int(rank)))#'cfg.model_dir = ./Pt/'     
        #     test(global_model, n_epi)# if n_epi%50 == 0:
        #     print('i is',n_epi)
        #     plt.plot(np.cumsum(total_reward_list))
            
        #     plt.xlabel('step')
        #     plt.ylabel('reward sum')
        # # plt.xlim(0,50)
        #     plt.ylim(-5,10)
        #     plt.title("train %dth episodes value_sum = %f, value_over_sum = %f"%(n_epi, sum_val,np.sum(envs.item_val) ))      
        #     plt.show()
        if (n_epi+1)%(max_episodes) == 0 :
            torch.save(global_model.state_dict(),  './Pt/a3c_train_%s_item_%d_knap_%d_epi_%d_rank_%d_epi_%d_act.pt'%(date,  global_model.itemsize, global_model.knapsize, max_episodes, int(rank) ,int(n_epi)))#'cfg.model_dir = ./Pt/'     
            data = {
                'lList': lList,
                'rList': rList,
                'sList': sList,
                'learning_rate': learning_rate,
        
        
            }         
        
            with open('train_%s_item_%d_knap_%d_epi_%d_data.pickle_'%(date,  global_model.itemsize, global_model.knapsize, int(n_epi)), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
def test(global_model, n_epi):
    print("hi")
    test_file = ''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rList = []
    sList = []
    idx = []
    last_state = []
    test_model  = ActorCritic(input_size, num_of_ac, N, kc)
    test_model.load_state_dict(global_model.state_dict())
    test_model = test_model.to(device)
    envs = env.ENV(N,kc, reward_op, state_op, test_file)

    max_epi = envs.overall_item_weight.shape[0]
    # t1 = time()
    for i in range(max_epi):
        done = False
        s = envs.reset(i)
        total_reward = 0.0;  
        total_reward_list = []
        # selected_list = []
        step_count = 0
        s_lst, a_lst, r_lst = [], [], []
        while not done:
            s2 = np.expand_dims(s, 1)
            prob = test_model.pi(torch.from_numpy(s2).double(), device)
            a = Categorical(prob.cpu()).sample().numpy()
            s_prime, r, done = envs.step(a)

            # selected_list.append(envs.selected*envs.item_val)  
            s = s_prime
            step_count += 1
            total_reward +=r
            total_reward_list.append(r)
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
        idx.append(envs.idx)        
        last_state.append(envs.selected)
        # knap_map.append(envs.knap_map)
        rList.append(total_reward)
        sList.append(step_count)   
    # if (i+1)%max_epi == 0:
    # t2 = time()
    # compu_time = t2 - t1
    data = {
        'rList': rList,
        'sList': sList,
        'learning_rate': learning_rate,
        # 'compu_time' : compu_time,
        # 'knap_map' : knap_map,
        'last_state': last_state,
        'idx' : idx
    }         
    # print(sum(selected_list))
    with open('a3c_knap_real_knap_1_%s_item_%d_knap_%d_data_%d_epi.pickle_'%(date,  global_model.itemsize, global_model.knapsize, n_epi), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # test_model = test_model.to("cpu")  
    test_model.load_state_dict(global_model.state_dict())
        # test_model = test_model.to(device)    

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    global_model  = ActorCritic(input_size, num_of_ac, N, kc)
    if load_op == 1:
        name = 'a3c_512_cpu_step_20end_0129_01_35_item_50_knap_3_epi_1000_rank_1_epi_19999_act.pt'
        global_model.load_state_dict(torch.load('./Pt/' + name))
    # global_model = ActorCritic()
    global_model.share_memory()
 
    processes = []
# use just call train in a2c
# train(global_model,0)
    for rank in range(n_train_processes):
            p =  mp.Process(target=train, args=(global_model, rank,))
              
    
            p.start()
            processes.append(p)
    for p in processes:
        p.join()