# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 22:13:34 2022

@author: gwsur
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:42:04 2022

@author: gwsur
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 07:48:44 2021

@author: gwsur
"""

import numpy as np
import math
import random
import pickle
import copy
from datetime import datetime
date = datetime.now().strftime('%m%d_%H_%M')
class ENV:

    #print(item_val[1])
    

    knap_capa = 3;
    learning_rate = 1e-1
#    input_size = a.shape[0]
    output_size = 1
    
    rList= []

    def __init__(self,n,k, reward_op, state_op, file, name="main"):
        self.net_name = name

        self.N = n 
        self.K = k
        self.reward_op = reward_op
        self.state_op = state_op
        # name = input("please write file name to open: (source item)")
        with open(file,'rb') as f:
            data = pickle.load(f)
        self.name = file
        self.overall_item_value = np.array(data.get('value'))
        self.overall_item_weight = np.array(data.get('weight'))
        self.overall_knap_capa = np.array(data.get('knapsack'))     
        self.ovrl_max = max(np.max(self.overall_item_value, axis=-1))
    def build_stat(self,i):
        # with tf.compat.v1.variable_scope(self.net_name):
            sn = random.randint(1, self.N) # select n sn <=N
       
            k = random.sample(range(0, self.N), sn) #choose item to problem instance 0 ~ N-1
            self.idx = np.asarray(k, dtype='int64')
            self.capa_idx = np.asarray(np.arange(self.K), dtype='int64')
            item_v = self.overall_item_value[i,self.idx]
            item_w = self.overall_item_weight[i,self.idx]
            problems_ratio = item_v.copy()/item_w.copy()

            sort_index = np.argsort(-problems_ratio)            
            self.sort_index = sort_index
            self.sorted_item_val = item_v[self.sort_index]
            self.sorted_item_weight= item_w[self.sort_index]
            self.item_size = self.sorted_item_weight.shape[0] 
         
            # self.item_weight = np.asarray(item_w)           
            # self.item_val = np.asarray(item_v)

             
            self.num_of_ac = self.N
            # self.knap_capa = self.problems[i,2,0]
            self.knap_capa = self.overall_knap_capa[i,self.capa_idx]
            self.knap_capa = np.random.randint(min(self.sorted_item_weight), int(max(self.knap_capa))+1, size=(self.K))

    
    def step(self, action):
        state = self.curr_state.copy()
        # state = self.curr_agre_state.copy()
        done = False
       
        k = 0 #math.floor(action/self.item_size)
        #capa 0 인 가방에 담으려 했을 때 
        # if state[0,1 + k] == 0:
        #     return state, - sum(self.curr_capa), False
        #k_idx = np.nonzero(self.capa_idx == self.curr_capa_idx[0,k])[0]
        k_idx2 = self.curr_capa_idx[0,k]
        
     #   item = action

# 담았던 아이템을 또 담으려 했을 때
        if action > state[0,0] - 1:

                # print('you tried to do again %d, %d'%(state[0,0],item+1))
            return self.curr_state.copy(), - 0.02 , False
        item_idx = self.curr_idx[0,action].copy() #np.nonzero(self.idx == self.curr_idx[0,action])[0]
        # lefted = self.selected.copy() + self.passed.copy()
        # lefted[item_idx] = 1



        temp_s = state.copy()

      #  selected_item = item_idx

        
        wFlag = False
        # 담으려고 시도한 아이템의 무게가 초과될때
        if state[0,1 + k] < self.sorted_item_weight[item_idx]:
            wFlag = True
            self.sorted_passed[item_idx] = 1
            # if self.reward_op == 0 or self.reward_op == 3: 
            #    return state, - state[0,self.N + 3 + self.K + item] , False
            # elif self.reward_op == 6 or self.reward_op == 1:
            #     return state, - self.item_weight[selected_item]/state[0, 1 + k] , False
            # # else:
            # #    return state, - self.item_weight[selected_item] , False
            # else:
            #    return state, - self.item_weight[selected_item] , False

        #안되는 케이스 끝나고 되는 케이
        temp_s[0,0] = temp_s[0,0] - 1
            
        if not wFlag:
            self.sorted_selected[item_idx] = 1
            self.curr_capa[k_idx2] = self.curr_capa[k_idx2] - self.sorted_item_weight[item_idx]

        
        # valid_k = np.nonzero(self.curr_capa > 0)
        
        # valid_k = np.asarray(valid_k)
        # valid_k = valid_k.reshape(1,valid_k.size)
        valid_k = np.argwhere(self.curr_capa > 0).T
        
        if valid_k.size == 0:
            capa = np.asarray(np.zeros(self.K))
         
        else:
          
            capa_ratio = self.curr_capa[valid_k]
            valid_k_sort =  np.argsort(-capa_ratio)
    
            # valid_k_sort = np.asarray(valid_k_sort)
            # valid_k_sort = valid_k_sort.reshape(1,valid_k_sort.size)
            capa = self.curr_capa[valid_k[0,valid_k_sort]]
            
            if valid_k.size < self.K:
                pad = np.asarray(np.zeros( self.K- valid_k.size))
                capa = np.append(capa, pad)  
            # capa = np.asarray(capa)    

        temp_s[0,1:1 + self.K] = capa
        
        lefted = self.sorted_selected.copy() + self.sorted_passed.copy()
        # lefted[item_idx] = 1
        # unSelected = np.nonzero(lefted == 0 )
        unSelected = np.argwhere(lefted == 0).reshape(-1)
        # if self.reward_op == 0:
        reward = self.sorted_item_val[item_idx]/self.ovrl_max #/100
        
        if unSelected.size == 0:
             temp_s[0,1 + self.K] = 0
             temp_s[0,2 + self.K] = 0
             temp_s[0, 3 + self.K: 3 + self.K + self.N] = 0
             temp_s[0, 3 + self.K + self.N: 3 + self.K + 2*self.N] = 0
             
             self.curr_idx = 0
             self.curr_state = temp_s.copy() 
             return temp_s, reward, True

        temp_s[0,1 + self.K] = sum(self.sorted_item_val[unSelected])
        temp_s[0,2 + self.K] = sum(self.sorted_item_weight[unSelected])
        
     #   problems_ratio = self.item_val[unSelected]/self.item_weight[unSelected]

      #  unSelected_sort = np.argsort(-problems_ratio)
        
        # unSelected = np.asarray(unSelected)
      #  unSelected = unSelected.reshape(1,unSelected.size)
     

     
        temp_s[0, 3 + self.K : 3 + self.K + 2*self.N] = np.zeros(2*self.N)
        temp_s[0, 3 + self.K : 3 + self.K + 2*unSelected.size:2] = self.sorted_item_val[unSelected]/self.ovrl_max
        if max(self.curr_capa) > 0:
            temp_s[0, 4 + self.K : 4 + self.K + 2*unSelected.size:2] = self.sorted_item_weight[unSelected]/max(self.curr_capa)
            self.curr_capa_idx = self.capa_idx[valid_k[0,valid_k_sort]]   
        else:
            temp_s[0, 4 + self.K : 4 + self.K + 2*unSelected.size:2] = self.sorted_item_weight[unSelected]
            done = True
            self.curr_capa_idx = 0
        # reward 0 and same state for non suitable item left
        
        self.curr_state = temp_s.copy() 
        
        self.curr_idx = unSelected
        self.curr_idx = self.curr_idx.reshape(1,unSelected.size)
        
        if wFlag:
            # if self.reward_op < 3 :
            #     return temp_s, - 0.1, False #self.item_weight[selected_item]/self.curr_capa[k_idx] 

            # else:
           return temp_s, - 0.001, False
           
        if max(self.curr_capa) < min(self.sorted_item_weight[unSelected]):
        # if temp_s[0,1] < min(self.item_weight[unSelected[0]]):
            done = True
        # capa full episode should stop

        # elif temp_s[0,0] == 0:
        #     done = True
    
        return temp_s, reward, done
    
    def reset(self,i):
        self.build_stat(i)
        

        # normal_capa =  np.expand_dims(self.knap_capa, axis=-1)
        
        k_sort_index = np.argsort(-self.knap_capa)
        capa_sorted = self.knap_capa[k_sort_index]
        normal_capa = capa_sorted
        normal_capa = np.asarray(normal_capa)
        # normal_val = []
        # t_mv = max(item_val_sorted/item_weight_sorted)

     

            

        arr1 = np.zeros(2*self.N+3 + self.K)
        arr1[0]= self.item_size
        arr1[1:self.K + 1] = normal_capa
        arr1[self.K + 1] = sum(self.sorted_item_val)
        arr1[self.K + 2] = sum(self.sorted_item_weight)
        arr1[self.K + 3:self.K + 3+2*self.item_size:2] = self.sorted_item_val/self.ovrl_max
        arr1[self.K + 4:self.K + 4+2*self.item_size:2] = self.sorted_item_weight/max(self.knap_capa)
        arr1[self.K + 4+2*self.item_size:self.K + 4+2*self.N] = 0
        arr1 = np.array(arr1)
        arr1 = arr1.reshape(1,arr1.size) 
        self.curr_capa_idx = np.asarray(copy.deepcopy(self.capa_idx))
        self.curr_capa_idx = self.curr_capa_idx[k_sort_index]
        self.curr_capa_idx = self.curr_capa_idx.reshape(1,self.K)
        self.curr_idx = np.asarray(np.arange(self.item_size), dtype='int64').reshape(-1)  #np.asarray(copy.deepcopy(self.idx))
        # self.curr_idx = self.curr_idx[sort_index]
        self.curr_idx = self.curr_idx.reshape(1,self.item_size)
        self.curr_state = arr1.copy()
        self.selected = np.zeros(self.item_size)
        self.sorted_selected = np.zeros(self.item_size)
        self.sorted_passed = np.zeros(self.item_size)
        self.curr_capa = copy.deepcopy(self.knap_capa)
        return arr1

