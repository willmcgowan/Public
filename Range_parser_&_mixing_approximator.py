#!/usr/bin/env python
# coding: utf-8

# In[220]:


import numpy as np
import pandas as pd
import random 

def range_parser(long_string):
    array = np.zeros([13,13])
    lis = long_string.split(',')
    setting = 1
    for i in range(len(lis)):
        if len(lis[i])==16:
            setting = float(lis[i][1:5])/100
            card= lis[i][6:9]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
            setting = 1
        elif len(lis[i])==15:
            setting = float(lis[i][1:5])/100
            card= lis[i][6:8]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
            setting = 1
        elif len(lis[i])==9 and lis[i].find('/')==-1:
            setting = float(lis[i][1:5])/100
            card = lis[i][6:9]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
        elif len(lis[i])==8:
            setting = float(lis[i][1:5])/100
            card = lis[i][6:8]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
        elif len(lis[i])==10:
            card = lis[i][0:3]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
            setting = 1
        elif len(lis[i])==9 and lis[i].find('/')!=-1:
            card = lis[i][0:2]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
            setting = 1
        elif len(lis[i])==7:
            var = hyphen_to_intervals(lis[i])
            if var[3]=='o':
                for j in range(var[1],var[2]+1):
                    array[j,var[0]]=setting
            else:
                for j in range(var[1],var[2]+1):
                    array[var[0],j]=setting 
        elif len(lis[i])==5:
            var = hyphen_to_intervals(lis[i])
            for j in range(var[0],var[1]+1):
                array[j,j]=setting
        elif len(lis[i])==3:
            card = lis[i][0:3]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
        elif len(lis[i])==2:
            card = lis[i][0:2]
            coord = card_to_coord(card)
            array[coord[0],coord[1]]=setting
    return array        
def card_to_coord(card):
    dict_ = {'A':0,'K':1,'Q':2,'J':3,'T':4,'9':5,'8':6,'7':7,'6':8,'5':9,'4':10,'3':11,'2':12}
    if len(card)==3:
        if card[2]=='o':
            return [dict_[card[1]],dict_[card[0]]]
        else:
            return [dict_[card[0]],dict_[card[1]]]
    else:
        return [dict_[card[0]],dict_[card[1]]]

def hyphen_to_intervals(str_):
    dict_ = {'A':0,'K':1,'Q':2,'J':3,'T':4,'9':5,'8':6,'7':7,'6':8,'5':9,'4':10,'3':11,'2':12}
    if len(str_) == 5:
        lower = dict_[str_[0]]
        upper = dict_[str_[3]]
        return [lower,upper]
    if len(str_)==7:
        root = dict_[str_[0]]
        type_ = str_[2]
        upper = dict_[str_[1]]
        lower = dict_[str_[5]]
        return [root,upper,lower,type_]
    
        
        


# In[240]:


def cooling_schedule(k,k_max):
    T_top = 10**6
    return k*(0-10**6)/k_max
def transition_function(E_new,E_previous,T):
    k_b = 1.38*(10**(-23))
    if E_new<E_previous:
        return 1
    elif E_new>E_previous:
        if np.random.uniform(0,1)<np.exp((-E_new+E_previous)/(k_b*T)):
            return 1
        else:
            return 0 
def find_available(num,abstraction):
    lis = [i/abstraction for i in range(abstraction+1)]
    if num in lis:
        return [num]
    for i in range(len(lis)):
        if lis[i]>num:
            return [lis[i-1],lis[i]]
def comb_matrixer():
    comb_matrix = np.zeros([13,13])
    for i in range(13):
        for j in range(13):
            if i>j:
                comb_matrix[i,j]=12
            if i<j:
                comb_matrix[i,j]=4
            if i==j:
                comb_matrix[i,j]=6
    return comb_matrix

def simulated_annealing(max_iters,init_range,action_range,ground_truth_frequencies,abstraction):
    comb_matrix = comb_matrixer()
    s = action_range.copy()
    s_0 = action_range.copy()
    entries = np.argwhere(s_0)
    for k in range(max_iters):
        T = cooling_schedule(k,max_iters)
        #randomly select an entry#
        i,j = list(random.choice(entries))
        lis = find_available(s_0[i,j],abstraction)
        choice=random.choice(lis)
        ground = comb_matrix*ground_truth_frequencies
        current = comb_matrix*init_range*s
        new_val = comb_matrix[i,j]*init_range[i,j]*choice
        old_val = comb_matrix[i,j]*init_range[i,j]*s[i,j]
        #needs modifying this deletes all low frequency bluffs#
        matrix = np.abs(current)
        useful_val=np.sum(matrix)-np.sum(ground)
        H_old = np.abs(useful_val)
        useful_val+= -matrix[i,j]+choice*init_range[i,j]*s[i,j]
        H_new = np.abs(useful_val)
        if transition_function(H_new,H_old,T)==1:
            s[i,j]=choice
        else:
            continue
    return s

    
    
        
        
        
        
        
        


# In[ ]:





# In[ ]:





# In[244]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




