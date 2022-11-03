# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:10:54 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:19:28 2021

@author: User
"""
import pickle
import numpy as np
import sys
import math
from datetime import datetime
date = datetime.now().strftime('%y%m%d_%H_%M')
n_ep = sys.argv[1]
N = sys.argv[2]
kn_c = sys.argv[3]
R = sys.argv[4]
R2 = sys.argv[5]
div = 1 #sys.argv[6]
mul = 1 #sys.argv[7]
# weight_mul = sys.argv[6]
v = np.random.randint(1, int(R)+1, size=(int(n_ep),int(N)))
w = np.random.randint(1, int(R)+1, size=(int(n_ep),int(N)))
v = w*(10-w)
k2 = np.random.randint(math.ceil(int(R2)/int(div)), int(mul)*int(R2)+1, size=(int(n_ep),int(kn_c))) 
data = {
    'n_ep': n_ep,
    'value': v,
    'weight': w,
    'knapsack': k2

}
# print( max(np.max(v, axis=-1)))
# print( max(np.max(k2, axis=-1)))
with open('curve_ep_%d_item_%d_knap_%d_R_%d_R2_%d_data.pickle_%s'%( int(n_ep), int(N), int(kn_c), int(R), int(R2), date), 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)   
print('curve_ep_%d_item_%d_knap_%d_R_%d_R2_%d_data.pickle_%s'%( int(n_ep), int(N), int(kn_c), int(R), int(R2), date), 'made! ')