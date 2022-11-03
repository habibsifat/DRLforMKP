# DRLforMKP
A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

This project shows the official codes that used in A Deep Reinforcement Learning-Based Scheme for Solving Multiple Knapsack Problems

Appl. Sci. 2022, 12(6), 3068; https://doi.org/10.3390/app12063068



![image](https://user-images.githubusercontent.com/69515626/199708217-af268d7a-d9eb-4502-979b-0aa87880aca7.png)
<Figure in the paper>

I used it in spyder IDE, and the scripts are as follow
creating item and knapsack instances

runfile('C:/yourdirectory/RI.py', wdir='C:/yourdirectory',args='1000 50 3 10 80')
runfile('C:/yourdirectory/LI.py', wdir='C:/yourdirectory',args='1000 50 3 10 10')
runfile('C:/yourdirectory/QI.py', wdir='C:/yourdirectory',args='1000 50 1 10 20')

train and test  (in here, the train file should be hard coded in a3c mode)

runfile('C:/yourdirectory/train.py', wdir='C:/yourdirectory',args='1000 0.0001 50 1000 5 0.9999999 6 4 0')
runfile('C:/yourdirectory/test.py', wdir='C:/yourdirectory',args='1000 0.0001 50 1 5 0.9999999 6 4')

comparison algorithm
To run, gurobi, you need a license

runfile('C:/yourdirectory/random_sol_knap.py', wdir='C:/yourdirectory',args='1000 0.001 50 1 1 0.99')
runcell(0, 'C:/yourdirectory/gurobi_op_mul.py')
runfile('C:/yourdirectory/ffh_mul.py', wdir='C:/yourdirectory',args='1000 0.001 50 1 1 0.99')

I will delete the redundant part ASAP, but the code works well in here.
