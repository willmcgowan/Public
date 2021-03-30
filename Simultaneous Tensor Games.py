#!/usr/bin/env python
# coding: utf-8

# In[51]:

import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

def payoff_constructor(n_players,choices_vect,construction_func):
    arr = np.zeros(choices_vect+[n_players])
    for p in itertools.product(*[[i for i in range(j)]for j in choices_vect]):
        arr[p]=construction_func(n_players)
    return arr
    
def sym_dirichlet(n):
    return np.random.dirichlet([1 for i in range(n)])

def two_zero(n):
    choice = random.sample([i for i in range(n)],2)
    var = np.zeros(n)
    var[choice[0]]=1
    var[choice[1]]=-1
    return var


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
    var = normalise(var)
    return var



def tensor_grad_descent(payoff_tensor,init_strategies,iters,gamma,noise):
    print("Init Expected Utility:")
    strategies = init_strategies
    n_players = len(strategies)
    for i in range(n_players):
        strategies[i]=normalise(strategies[i])
    print(expectation(payoff_tensor,strategies))
    history = [[np.zeros(len(init_strategies[j]))for j in range(n_players)] for k in range(iters+1)]
    history[0]=strategies
    for iteration in range(iters):
        new_strategies = [np.zeros(len(strategies[j])) for j in range(n_players)]
        for player in range(n_players):
            new_strategies[player]=evolve(payoff_tensor,strategies,player,gamma)+noise*((iters-iteration)/(iters))*np.random.dirichlet([1 for i in range(len(new_strategies[player]))])#dirichlet noise just because the constraints sometimes cause the gradients to not produce a useful direction
            new_strategies[player]=normalise(new_strategies[player])
            history[iteration+1][player]=new_strategies[player]
        strategies = new_strategies
    print("End Expected Utility:")
    print(expectation(payoff_tensor,strategies))
    print('End Strategies:')
    print(strategies)
    return history





def plotter(history):
    player_strategies = [[history[i][player] for i in range(len(history))]for player in range(len(history[0]))]
    colors = ['r','g','b','y','m','k']
    fig, axs = plt.subplots(len(history[0]))
    fig.suptitle("Orbits")
    for i in range(len(history[0])):
        axs[i].scatter([player_strategies[i][j][0] for j in range(len(history))],[player_strategies[i][j][1] for j in range(len(history))],color=colors[i])
        axs[i].set_xlim(0,1)
        axs[i].set_ylim(0,1)
    return None
    



