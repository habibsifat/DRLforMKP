# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:06:03 2021

@author: User
"""

import numpy as np
import random
import pickle
import copy
import sys
import torch
from datetime import datetime
from time import time
date = datetime.now().strftime('%m%d_%H_%M')



def md1_delay(mu,lamda):

    rho = (lamda/mu)
    if rho >= 1:
        return 1
    # print('rho',rho)
    delay = rho/(2*mu*(1-rho)) + 1/mu
    return delay



def main(max_episodes, learning_rate, N, R, kc, gamma):   

    max_episodes = int(max_episodes)
    learning_rate = float(learning_rate)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N = int(N)

    kc = int(kc)
    gamma = float(gamma)

    name = input("please write file name to open: ")
    with open(name,'rb') as f:
        data = pickle.load(f)
    
    overall_item_value = np.array(data.get('value'))
    overall_item_weight = np.array(data.get('weight'))
    overall_knap_capa = np.array(data.get('knapsack'))    
    last_state = []
    knap_map = []
    s_knap_map = []
    t1 = time()
    for i in range(max_episodes): 
    
        # step_count = 0
    

        # knap_map = np.zeros((overall_knap_capa[i].copy().size,overall_item_value[0].shape[0]))
        # knap_capa = overall_knap_capa[i].copy()
        capa = overall_knap_capa[i].copy()
        # dreq = np.ones(kc)
        # item_value = overall_item_value[i].copy()
        # item_weight = overall_item_weight[i].copy()

        last_state_ca = []
        knap_map_ca = []
        sur_value = []
        for m in range(1000):
            capa = overall_knap_capa[i].copy()
            sn = random.randint(1, N) # select n sn <=N
       
            k = random.sample(range(0, N), sn) #choose item to problem instance 0 ~ N-1
            selected = np.zeros((1,N))   
            # selected[0,k] = 1
            for j in range(sn):
                k_idx = random.randint(0,kc-1)
                if capa[k_idx] - overall_item_weight[i,k[j]] >= 0:
                    capa[k_idx] = capa[k_idx] - overall_item_weight[i,k[j]]
                    selected[0,k[j]] = 1
            last_state_ca.append(selected)
            knap_map_ca.append(selected) 


            sv = np.sum(selected*overall_item_value[i],axis=1)
            sur_value.append(sv)
        a = np.argmax(sur_value)
        last_state.append(last_state_ca[a])
    t2 = time()
    computime = t2 - t1
    data = {

        'last_state' : last_state,
        'compute_time':computime
    }
    with open('random_sol_knap_end_%s_item_%d_knap_%d_data.pickle_'%(date,  N, kc), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) 
        
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5], sys.argv[6])