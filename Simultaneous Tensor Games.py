#!/usr/bin/env python
# coding: utf-8

# In[177]:


from numba import njit,jit
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import cProfile
import collections
import torch


# In[536]:



def payoff_constructor(n_players,choices_vect,construction_func):
    arr = np.zeros(choices_vect+[n_players])
    for p in itertools.product(*[[i for i in range(j)]for j in choices_vect]):
        arr[p]=construction_func(p,n_players)
    return arr

def sym_dirichlet(p,n):
    return np.random.dirichlet([1 for i in range(n)])-np.array([1/n for i in range(n)])

def two_zero(p,n):
    choice = random.sample([i for i in range(n)],2)
    var = np.zeros(n)
    var[choice[0]]=1
    var[choice[1]]=-1
    return var

def n_player_rochambo(p,n):
    p = np.array(list(p))
    count = [np.count_nonzero(p==i) for i in range(n)]
    if count[0]>0 and count[1]>0 and count[2]>0:
        return np.zeros(n)
    elif count[0]==n or count[1]==n or count[2]==n:
        return np.zeros(n)
    elif count[0]==0:
        return np.array([count[1]/count[2] if i ==2 else -1 for i in p])
    elif count[1]==0:
        return np.array([count[2]/count[0] if i==0 else -1 for i in p])
    elif count[2]==0:
        return np.array([count[0]/count[1] if i==1 else -1 for i in p])

def expectation(payoff_tensor,strategies):
    result = np.zeros(payoff_tensor.shape[-1])
    for p in itertools.product(*[[i for i in range(j)]for j in payoff_tensor.shape[:-1]]):
        var = 1
        for player in range(len(strategies)):
            var*=strategies[player][p[player]]
        for player in range(len(strategies)):
            result[player]+= var*payoff_tensor[p][player]
    return result
        
    
def grad(payoff_tensor,strategies,player):
    res = np.zeros(len(strategies[player]))
    for i in range(len(strategies[player])):
        for p in itertools.product(*[[j for j in range(payoff_tensor.shape[k])] if k!=player else [i] for k in range(len(payoff_tensor.shape[:-1]))]):
            var = 1
            for q in range(len(strategies)):
                if q!=player:
                    var*=strategies[q][p[q]]
                else:
                    var*=1
            res[i]+=var*payoff_tensor[p][player]
    return res

def evolve(payoff_tensor,strategies,player,gamma):
    var = strategies[player]+gamma*grad(payoff_tensor,strategies,player)
    for i in range(var.shape[0]):
        var[i]= max(var[i],0)
    var = vector_normalise(var)
    return var

def tensor_grad_descent(payoff_tensor,init_strategies,iters,gamma,noise):
    print("Init Expected Utility:")
    strategies = init_strategies
    n_players = len(strategies)
    for i in range(n_players):
        strategies[i]=vector_normalise(strategies[i])
    print(expectation(payoff_tensor,strategies))
    history = [[np.zeros(len(init_strategies[j]))for j in range(n_players)] for k in range(iters+1)]
    history[0]=strategies
    for iteration in range(iters):
        new_strategies = [np.zeros(len(strategies[j])) for j in range(n_players)]
        for player in range(n_players):
            new_strategies[player]=evolve(payoff_tensor,strategies,player,gamma)+noise*((iters-iteration)/(iters))*np.random.dirichlet([1 for i in range(len(new_strategies[player]))])
            new_strategies[player]=vector_normalise(new_strategies[player])
            history[iteration+1][player]=new_strategies[player]
        strategies = new_strategies
    print("End Expected Utility:")
    print(expectation(payoff_tensor,strategies))
    print('End Strategies:')
    print(strategies)
    return history


def vector_normalise(v):
    return v/v.sum()

def plotter(history):
    player_strategies = [[history[i][player] for i in range(len(history))]for player in range(len(history[0]))]
    colors = ['r','g','b','y','m','k']
    fig, axs = plt.subplots(len(history[0]),figsize = (15,15))
    fig.suptitle("Orbits")
    for i in range(len(history[0])):
        axs[i].scatter([player_strategies[i][j][0] for j in range(len(history))],[player_strategies[i][j][1] for j in range(len(history))],color=colors[i],s=2)
        axs[i].set_xlim(0,1)
        axs[i].set_ylim(0,1)
    return None
    
def Jacobian(payoff_tensor,strategies):
    n_players = strategies.shape[0]
    n_choices = strategies.shape[1]
    jacobian= np.zeros([n_players*n_choices,n_players*n_choices])
    for i in range(n_players):
        for j in range(n_players):
            for k in range(n_choices):
                for l in range(n_choices):
                    jacobian[i*n_choices+k,j*n_choices+l]=d2_func(payoff_tensor,strategies,i,j,k,l)
    return jacobian
                

def d2_func(payoff_tensor,strategies,i,j,k,l):
    var = copy.deepcopy(strategies)
    var[i,:]= np.zeros([var[i].shape[0]])
    var[i,k]=1.0
    var[j,:]= np.zeros([var[j].shape[0]])
    var[j,l]=1.0
    tensor = np.multiply.reduce(np.ix_(*var))
    prod = np.tensordot(tensor,payoff_tensor,strategies.shape[0])[i]
    return prod
    

def SymplecticGradientAdjustment(payoff_tensor,strategies,epsilon):
    simul_grad = np.ravel(SimultaneousGrad(payoff_tensor,strategies))
    jacobian = Jacobian(payoff_tensor,strategies)
    antisymm = (jacobian-jacobian.transpose())/2
    adj =  np.dot(antisymm,simul_grad)
    dH = np.dot(jacobian.transpose(),simul_grad)
    align = np.sign(np.dot(dH,simul_grad)*np.dot(adj,dH)+epsilon)
    return simul_grad + align*adj

#clear each player must have the same num of choices a tad cringe
def SimultaneousGrad(payoff_tensor,strategies):
    n_players = strategies.shape[0]
    n_choices = strategies.shape[1]
    simul_grad = np.zeros([n_players,n_choices])
    for i in range(n_players):
        for j in range(n_choices):
            strat_copy = copy.deepcopy(strategies)
            strat_copy[i,:]=np.zeros(n_choices)
            strat_copy[i,j]=1
            tensor = np.multiply.reduce(np.ix_(*strat_copy))
            prod = np.tensordot(tensor,payoff_tensor,n_players)[i]
            simul_grad[i,j]=prod
    return simul_grad
    
def Optimiser(payoff_tensor,init_strategies,learning_rate,iters,noise_param):
    n_players = init_strategies.shape[0]
    n_choices = init_strategies.shape[1]
    history = np.zeros([iters,n_players,n_choices])
    strategies = normalize(init_strategies)
    for i in range(iters):
        for j in range(n_players):
            history[i,j]=strategies[j,:]
        noise = noise_param*np.random.dirichlet([1 for k in range(n_players*n_choices)])
        for j in range(n_players):
            strategies[j,:]=history[i,j,:]
        adj = SymplecticGradientAdjustment(payoff_tensor,strategies,0.1)
        delta = adj.reshape([n_players,n_choices])
        strategies= strategies +learning_rate*delta+noise.reshape([n_players,n_choices])*(iters-i)/iters
        strategies = normalize(strategies)
    return history
    
def normalize(matrix):
    bools = np.heaviside(matrix,0)
    matrix = matrix*bools
    sums = matrix.sum(axis=1)
    for i in range(matrix.shape[0]):
        if sums[i]==0:
            sums[i]=1
            matrix[i,:]=np.random.dirichlet([1 for i in range(matrix.shape[1])])
    newsum = np.array([np.array([sums[i] for j in range(matrix.shape[1])])for i in range(matrix.shape[0])])
    return matrix/newsum
    
        
    
        
        

    






