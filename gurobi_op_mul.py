# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:47:28 2021

@author: User
"""


from __future__ import print_function
import gurobipy as gp
from gurobipy import GRB
import sys
import pickle
import numpy as np
from datetime import datetime
date = datetime.now().strftime('%m%d_%H_%M')
from time import time
try:
# Sample data
 
    objCoef = [32 , 32 , 15 , 15 , 6 , 6 , 1 , 1 , 1 , 1]
    knapsackCoef = [16 , 16 , 8 , 8 , 4 , 4 , 2 , 2 , 1 , 1]
    file = 'curve_ep_1000_item_50_knap_5_R_10_R2_40_data.pickle_220207_19_16'
    with open(file,'rb') as f:
        data = pickle.load(f)
    name = file
    overall_item_value = np.array(data.get('value'))
    overall_item_weight = np.array(data.get('weight'))
    overall_knap_capa = np.array(data.get('knapsack'))     
    Groundset = range (overall_item_value.shape[1])
    Budget = 30
    Budget_lst = [30,30]
    last_state = []
    # Create initial model
    t1 = time()
    for i in range(overall_item_value.shape[0]):
        N = overall_item_value.shape[1]
        
        selected = np.zeros((1,overall_item_value.shape[1]))  
        values = overall_item_value[i]
        weights = overall_item_weight[i]
        capacities = overall_knap_capa[i]
        M = capacities.shape[0]
        t_selected = np.zeros((1,N*M))  
        
        # This values above are just exemples, my data is too bigger than them so I just simplified what I want to say #
        m = gp.Model()
        
        # Insert the decision variables
        x = m.addVars(N, M, vtype = gp.GRB.BINARY)
        # Define the objective function
        m.setObjective(gp.quicksum(x[k, j] * values[k] for k in range(N) for j in range(M)), sense=gp.GRB.MAXIMIZE)
        
        # Constraint 1
        c1 = m.addConstrs(gp.quicksum(x[k,j] for j in  range(M)) <= 1 for k in range(N))
        
        # Constraint 2 
        c2 = m.addConstrs(gp.quicksum(x[k,j] * weights[k] for k in range(N)) <= capacities[j] for j in  range(M))
        
        # Run the model
        m.optimize()
        t = 0
        if m.SolCount > 0:
             m.printAttr('X')
        #     m.printAttr('X')
        #     m.printAttr('Status')
        # print(m.getVars())
        for v in m.getVars():
            q = int(np.floor(t/M))
            # print(v)
            # print (v.VarName, v.X)
            selected[0,q] = selected[0,q] + int(v.X)
            # if int(v.X) > 0:
                # print(t)
            t = t+1 # 
            
        # print(t)

        # print(x)
        v = m.getVars()
        last_state.append(selected)
    t2 = time()
    computime = t2-t1
    data = {

        'last_state' : last_state,
         'compute_time':computime
    }
    with open('gurobi_op_sol_knap_end_%s_item_%d_knap_%d_data.pickle_'%(date,  overall_item_value.shape[1], overall_knap_capa.shape[1]), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
except gp.GurobiError as e :
    print ('Gurobi error ' + str( e.errno ) + ": " + str( e.message ))
except AttributeError as e :
    print ('Encountered an attribute error : ' + str ( e ))
