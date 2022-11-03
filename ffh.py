# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:13:17 2021

@author: User
"""



import numpy as np
import matplotlib.pyplot as plt
import random
import collections 
from torch.distributions import Categorical
import sys
import pickle 
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
date = datetime.now().strftime('%m%d_%H_%M')
# num_episodes = 3
# print('sys.argv 길이 : ', len(sys.argv))                                             
 
 
# for arg in sys.argv: 
#     print('arg value = ', arg) 


#num_of_st = pow(2,num_of_ac);



rList= []
sList = []

  
def rargmax(vector):
    
    m = np.amax(vector)
    indices = np.nonzero(vector == m )[0]
    return random.choice(indices)

# def knap_play(mainDQN, max_episodes, fla, step_max):
     
#      states = env3.ENV()
    
#      reward_sum = 0
#      step = 0
#      if fla == 0:
#          k = random.randint(0, max_episodes-1)
#          print("random k is {}".format(k))
#      else:
#          k = 190
#      state = states.reset(k)
#      while True:
         
#          action = np.argmax(mainDQN.predict(state))
#          state, reward, done  = states.step(state, action)
#          reward_sum += reward
#          step+=1
#          if step > step_max:
#              done = True         
#          if done:
#              print("state: {}".format(state))
#              print("Total score: {}".format(reward_sum))
#              break

         
         
# def main(max_episodes, greed_num, step_max, train_freq, sample_size, train_iter_freq, length, end_reward):
#     #max_episodes = 3
#     max_episodes = int(max_episodes)
#     greed_num = int(greed_num)
#     step_max = int(step_max)
def main(max_episodes, learning_rate, N, R, kc, gamma):   
    

    max_episodes = int(max_episodes)
    learning_rate = float(learning_rate)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N = int(N)

    kc = int(kc)
    gamma = float(gamma)
    

    name = input("please write file name to open: (source item)")
    
    with open(name,'rb') as f:
        data = pickle.load(f)

    overall_item_value = np.array(data.get('value'))
    overall_item_weight = np.array(data.get('weight'))
    overall_knap_capa = np.array(data.get('knapsack'))     
 
    

    last_state = []

    rList = []
    idx = []

    i = 0

    t1 = time()
    #Q = np.zeros([num_of_st, num_of_ac])
    for i in range(max_episodes): 

        step_count = 0

        selected = np.zeros((1,  N))
        capa = np.asarray(overall_knap_capa[i].copy()).reshape(1,overall_knap_capa[i].copy().size)
        desorder = np.asarray(np.argsort(-overall_item_value[i]/overall_item_weight[i])).reshape(1,overall_item_value[i].copy().size)
        if kc > 1:
            randknap = np.random.randint(0, int(kc), size=(1,int(N)))
        else:
            randknap = np.zeros((1,N)).astype(int)
        for j in range(N):
            k_desorder =  np.argsort(-capa)
            for k in range(kc):
                # print(k_desorder[0,k])
                # print(capa[0,k_desorder[0,k]])
                # print(desorder[0,j])
                # print(overall_item_weight[0,desorder[0,j]])
                if capa[0,k_desorder[0,k]] >= overall_item_weight[i,desorder[0,j]]:
                    capa[0,k_desorder[0,k]] = capa[0,k_desorder[0,k]] - overall_item_weight[i,desorder[0,j]]
                    selected[0,desorder[0,j]] = 1
                    break
           
        last_state.append(selected) 

    # plt.figure()
    # plt.plot(np.array(rList)/np.array(sList)) 
    # plt.show()
    t2 = time()

    computime = t2-t1
    data = {

        'last_state' : last_state,
        'compute_time':computime
    }
    with open('ffh_mul_end_%s_item_%d_knap_%d_data.pickle_'%(date,  N, kc), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)    
   # fla = 0
    # knap_play(mainDQN , max_episodes,fla, step_max)
    # fla = 1
    # knap_play(mainDQN , max_episodes,fla, step_max)
         
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5], sys.argv[6])
 #   main(50,7,5,7)
