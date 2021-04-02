
from numba import njit,jit
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import cProfile
import collections
import copy


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
    reform = np.array(strategies)
    tensor = np.multiply.reduce(np.ix_(*reform))
    result = np.tensordot(tensor,payoff_tensor,reform.shape[0])
    return result
  
def n_simplex_proj(vec):
    u = np.sort(vec)[::-1]
    D = u.shape[0]
    rho = 0
    for i in range(D):
        val= u[i]+(1/(i+1))*(1-np.sum(u[:i+1]))
        if val>0:
            rho =i+1
    mu = 1/rho*(1-np.sum(u[:rho]))
    return np.maximum(vec+mu*np.ones(D),np.zeros(D))

def jacobian(payoff_tensor,strategies):
    n_players = strategies.shape[0]
    n_choices = strategies.shape[1]
    jacob= np.zeros([n_players*n_choices,n_players*n_choices])
    for i in range(n_players):
        for j in range(n_players):
            for k in range(n_choices):
                for l in range(n_choices):
                    jacob[i*n_choices+k,j*n_choices+l]=d2_func(payoff_tensor,strategies,i,j,k,l)
    return jacob
                

def d2_func(payoff_tensor,strategies,i,j,k,l):
    var = copy.deepcopy(strategies)
    var[i,:]= np.zeros([var[i].shape[0]])
    var[i,k]=1.0
    var[j,:]= np.zeros([var[j].shape[0]])
    var[j,l]=1.0
    tensor = np.multiply.reduce(np.ix_(*var))
    prod = np.tensordot(tensor,payoff_tensor,strategies.shape[0])[i]
    return prod
    

def symplectic_grad_adj(payoff_tensor,strategies,epsilon):
    sim_grad = np.ravel(simul_grad(payoff_tensor,strategies))
    jacob = jacobian(payoff_tensor,strategies)
    antisymm = (jacob-jacob.transpose())/2
    adj =  np.dot(antisymm,sim_grad)
    dH = np.dot(jacob.transpose(),sim_grad)
    align = np.sign(np.dot(dH,sim_grad)*np.dot(adj,dH)+epsilon)
    return sim_grad + align*adj

#clear each player must have the same num of choices a tad cringe
def simul_grad(payoff_tensor,strategies):
    n_players = strategies.shape[0]
    n_choices = strategies.shape[1]
    sim_grad = np.zeros([n_players,n_choices])
    for i in range(n_players):
        for j in range(n_choices):
            strat_copy = copy.deepcopy(strategies)
            strat_copy[i,:]=np.zeros(n_choices)
            strat_copy[i,j]=1
            tensor = np.multiply.reduce(np.ix_(*strat_copy))
            prod = np.tensordot(tensor,payoff_tensor,n_players)[i]
            sim_grad[i,j]=prod
    return sim_grad
    
def optimiser(payoff_tensor,init_strategies,learning_rate,iters,noise_param):
    n_players = init_strategies.shape[0]
    n_choices = init_strategies.shape[1]
    history = np.zeros([iters,n_players,n_choices])
    strategies = np.zeros([n_players,n_choices])
    for i in range(n_players):
        strategies[i]=n_simplex_proj(init_strategies[i])
    for i in range(iters):
        for j in range(n_players):
            history[i,j]=strategies[j,:]
        noise = noise_param*np.random.dirichlet([1 for k in range(n_players*n_choices)])
        for j in range(n_players):
            strategies[j,:]=history[i,j,:]
        adj = symplectic_grad_adj(payoff_tensor,strategies,0.1)
        delta = adj.reshape([n_players,n_choices])
        strategies= strategies +learning_rate*delta+noise.reshape([n_players,n_choices])*(iters-i)/iters
        for k in range(n_players):
            strategies[k] = n_simplex_proj(strategies[k])
    return history

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
    
        
    
        
        

    






