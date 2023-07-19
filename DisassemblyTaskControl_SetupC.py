import gym
from gym import Env
from gym import core, spaces
from gym import utils
from gym.utils import seeding
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box
from random import *
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import stable_baselines3
from sb3_contrib import TRPO

from stable_baselines3.common import vec_env 
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
import numpy as np
import os
import argparse
import json
import math

from typing import Optional, Union

class InventoryEnv(gym.Env):
  def __init__(self, *args, **kwargs):
    #action = 0 corresponds to [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #action = 1 corresponds to [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #action = 2 corresponds to [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #action = 3 corresponds to [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    #action = 4 corresponds to [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    #action = 5 corresponds to [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    #action = 6 corresponds to [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    #action = 7 corresponds to [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    #action = 8 corresponds to [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    #action = 9 corresponds to [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    #action = 10 corresponds to [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    #action = 11 corresponds to [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    #action = 12 corresponds to [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    #action = 13 corresponds to [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    #action = 14 corresponds to [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    #action = 15 corresponds to [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    #action = 16 corresponds to [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.action_space = gym.spaces.Discrete(17)
    
    #Observation is the inventory and demand position at any given time
    #Requirement - 9 Inv - 12
    self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) , high = np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]), dtype=np.int64)

    # intialize
    self.reset()

    self.seed(10)

  def seed(self,seed=None):
    if seed != None:
      np.random.seed(seed=int(seed))

  
  def demand(self):
    #BEMOFCQLD

    return [np.random.poisson(self.demand_[0]), np.random.poisson(self.demand_[1]), np.random.poisson(self.demand_[2]), np.random.poisson(self.demand_[3]), np.random.poisson(self.demand_[4]) , np.random.poisson(self.demand_[5]) , np.random.poisson(self.demand_[6]) , np.random.poisson(self.demand_[7]) , np.random.poisson(self.demand_[8])]
  
  def transition(self, s, a, x):
    
    self.state_comp_old = self.state_comp

    #Generate new demand for components
    Demand = self.demand()

    #Update the total demand
    #BEMOFCQLD
    for i in range(9):
      self.tot_demand[i] += Demand[i]
    
    #New product arrivals
  
    arr = np.array([np.random.poisson(self.arrivals[0]) , np.random.poisson(self.arrivals[1]) , np.random.poisson(self.arrivals[2]), 
                    np.random.poisson(self.arrivals[3]), np.random.poisson(self.arrivals[4]), np.random.poisson(self.arrivals[5]) ,
                    np.random.poisson(self.arrivals[6]), np.random.poisson(self.arrivals[7]), np.random.poisson(self.arrivals[8]) , 
                    np.random.poisson(self.arrivals[9]), np.random.poisson(self.arrivals[10]), np.random.poisson(self.arrivals[11]) , 
                    np.random.poisson(self.arrivals[12]), np.random.poisson(self.arrivals[13]), np.random.poisson(self.arrivals[14]) , 
                    np.random.poisson(self.arrivals[15])])
    
    #Order - EMOLDQ
    
    occ = [0,0,0,0,0,0,0,0,0]
    
    if (self.WS_occupied[0][0] == True):
        occ[0] += 1
    if (self.WS_occupied[0][1] == True):
        occ[0] += 1
    if (self.WS_occupied[1][0] == True):
        occ[1] += 1
    if (self.WS_occupied[1][1] == True):
        occ[1] += 1
    if (self.WS_occupied[2][0] == True):
        occ[2] += 1
    if (self.WS_occupied[2][1] == True):
        occ[2] += 1
    if (self.WS_occupied[3][0] == True):
        occ[3] += 1
    if (self.WS_occupied[3][1] == True):
        occ[3] += 1
    if (self.WS_occupied[4][0] == True and self.WS_inprocess[4][0] ==[1,0]):
        occ[4] += 1
        occ[5] += 1
    if (self.WS_occupied[4][1] == True and self.WS_inprocess[4][1] ==[1,0]):
        occ[4] += 1
        occ[5] += 1
    if (self.WS_occupied[4][0] == True and self.WS_inprocess[4][0] ==[0,1]):
        occ[4] += 1
        occ[6] += 1
    if (self.WS_occupied[4][1] == True and self.WS_inprocess[4][1] ==[0,1]):
        occ[4] += 1
        occ[6] += 1
    if(self.WS_occupied[5][0] == True):
        occ[7] += 1 
    if(self.WS_occupied[5][1] == True):
        occ[7] += 1 
    if(self.WS_occupied[6][0] == True):
        occ[8] += 1
        occ[6] += 1
    if(self.WS_occupied[6][1] == True):
        occ[8] += 1
        occ[6] += 1
 
    #Get the amount of F and C in-process
    count_F = list(self.WS_queue[4].queue).count([1,0])
    count_C = list(self.WS_queue[4].queue).count([0,1])

    #BEMOFCQLD
    self.requirement[0] = max(self.tot_demand[0]-self.WS_queue[0].qsize()-x[20]-occ[0],0) #B 
    self.requirement[1] = max(self.tot_demand[1]-self.WS_queue[1].qsize()-x[21]-occ[1],0) #E 
    self.requirement[2] = max(self.tot_demand[2]-self.WS_queue[2].qsize()-x[22]-occ[2],0) #M
    self.requirement[3] = max(self.tot_demand[3]-self.WS_queue[3].qsize()-x[23]-occ[3],0) #O
    self.requirement[4] = max(self.tot_demand[4]-count_F-x[24]-occ[4],0) #F
    self.requirement[5] = max(self.tot_demand[5]-count_C-x[25]-occ[5],0) #C
    self.requirement[6] = max(self.tot_demand[6]-self.WS_queue[4].qsize()-self.WS_queue[6].qsize()-x[26]-occ[6],0) #Q  
    self.requirement[7] = max(self.tot_demand[7]-self.WS_queue[5].qsize()-x[27]-occ[7],0) #L  
    self.requirement[8] = max(self.tot_demand[8]-self.WS_queue[6].qsize()-x[28]-occ[8],0) #D

    
    self.WS_queue_copy = self.WS_queue

    ###########Workstation Anatomy
    #WS1 - 4mins , WS2 - 6mins, WS3 - 4mins, WS4 - 2mins, WS5 - 2mins
    release = [[[0,0,0],[0,0,0]] for i in range(7)]

    for i in range(7):
        for j in range(2):
      #Check if any current job will be ended in this timestep, if so, release
      #the current job and add new job or set the machine free.If not, if the machine is occupied, count time
            if (self.timestep[i][j] == self.processing_time[i]):
        
                release[i][j] = self.WS_inprocess[i][j]
                self.WS_inprocess[i][j] = ''
                self.WS_occupied[i][j] = False
                self.timestep[i][j] = 0

            if (self.WS_queue[i].empty() == False):
                self.WS_inprocess[i][j] = self.WS_queue[i].get()
                self.WS_occupied[i][j] = True
                self.timestep[i][j] = 1

            elif (self.WS_occupied[i][j] == True):
                self.timestep[i][j] += 1
  
    actions = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]
    
    WS_index = 0
    index_start = 0
    k=0
    repeat = [3,3,3,3,2,1,1]
    index_end = [3,6,9,12,14,15,16]

    for i in repeat:

      for j in range(i):
      
        if (a == actions[k]):
          #If the machine is occupied, add the incoming orders to 
          #the queue. If not, send one from the queue for disassembly, add any that is coming in to the queue
          if (self.WS_occupied[WS_index][0] == True and self.WS_occupied[WS_index][1] == True):
            self.WS_queue[WS_index].put(a[index_start:index_end[WS_index]])
          elif(self.WS_occupied[WS_index][0] == False and self.WS_occupied[WS_index][1] == True):
            self.WS_inprocess[WS_index][0] = a[index_start:index_end[WS_index]]
            self.WS_occupied[WS_index][0] = True
            self.timestep[WS_index][0] += 1
          elif(self.WS_occupied[WS_index][0] == True and self.WS_occupied[WS_index][1] == False):
            self.WS_inprocess[WS_index][1] = a[index_start:index_end[WS_index]]
            self.WS_occupied[WS_index][1] = True
            self.timestep[WS_index][1] += 1
          else:
            #If the queue is empty, send the incoming one to disassembly
            self.WS_inprocess[WS_index][0] = a[index_start:index_end[WS_index]]
            self.WS_occupied[WS_index][0] = True
            self.timestep[WS_index][0] += 1
        k += 1        
      
      
      index_start = index_end[WS_index]
      WS_index += 1

  

    #########Fulfil the demand 
    #BEMOFCQLD
    x_new = np.array([max(x[20]+sum(release[0][0])+sum(release[0][1])-self.tot_demand[0], 0),
                      max(x[21]+sum(release[1][0])+sum(release[1][1])-self.tot_demand[1], 0),
                      max(x[22]+sum(release[2][0])+sum(release[2][1])-self.tot_demand[2], 0),
                      max(x[23]+sum(release[3][0])+sum(release[3][1])-self.tot_demand[3], 0),
                      max(x[24]+sum(release[4][0])+sum(release[4][1])-self.tot_demand[4], 0),
                      max(x[25]+release[4][0][0]+release[4][1][0]-self.tot_demand[5], 0),
                      max(x[26]+release[4][0][1]+release[4][1][1]+sum(release[6][0])+sum(release[6][1])-self.tot_demand[6], 0),
                      max(x[27]+sum(release[5][0])+sum(release[5][1])-self.tot_demand[7], 0),
                      max(x[28]+sum(release[6][0])+sum(release[6][1])-self.tot_demand[8], 0)])


    self.demand_fulfilled = np.array([min(self.tot_demand[0],x[20]+sum(release[0][0])+sum(release[0][1])), 
                                      min(self.tot_demand[1],x[21]+sum(release[1][0])+sum(release[1][1])),
                                      min(self.tot_demand[2],x[22]+sum(release[2][0])+sum(release[2][1])),
                                      min(self.tot_demand[3],x[23]+sum(release[3][0])+sum(release[3][1])),
                                      min(self.tot_demand[4],x[24]+sum(release[4][0])+sum(release[4][1])),
                                      min(self.tot_demand[5],x[25]+release[4][0][0] + release[4][1][0] ),
                                      min(self.tot_demand[6],x[26]+release[4][0][1] + release[4][1][1]+sum(release[6][0])+sum(release[6][1])),
                                      min(self.tot_demand[7],x[27]+sum(release[5][0])+sum(release[5][1])),
                                      min(self.tot_demand[8],x[28]+sum(release[6][0])+sum(release[6][1]))])
   
    for i in range (9):
      self.tot_demand[i] -= self.demand_fulfilled[i]

    #state_comp = [ReB, ReE, ReM, ReO, ReF, ReC, ReQ, ReL, ReD, CF, QF, QD]
    ###########Update the state composition
    for i in range (13):
      if (x[i]>0):
        x[i] = x[i]-a[i+3]
      else:
        self.products[i+3] -=a[i+3]

      self.products[i+3] += arr[i+3]
    
    for i in range (3):
      self.products[i] = self.products[i] - a[i] + arr[i] 


    self.inventory = np.array([
                    max(x[0]+release[0][0][0]+release[0][1][0],0),
                    max(x[1]+release[0][0][1]+release[0][1][1],0),
                    max(x[2]+release[0][0][2]+release[0][1][2],0),
                    max(x[3]+release[1][0][0]+release[1][1][0],0),
                    max(x[4]+release[1][0][1]+release[1][1][1],0),
                    max(x[5]+release[1][0][2]+release[1][1][2],0),
                    max(x[6]+release[2][0][0]+release[2][1][0],0),
                    max(x[7]+release[2][0][1]+release[2][1][1],0),
                    max(x[8]+release[2][0][2]+release[2][1][2],0),
                    max(x[9]+release[3][0][0]+release[3][1][0],0),
                    max(x[10]+release[3][0][2]+release[3][1][2],0),
                    max(x[11]+release[3][0][1]+release[3][1][1],0),
                    max(x[12]+release[5][0][0]+release[5][1][0],0),
                    self.WS_queue[0].qsize(),
                    self.WS_queue[1].qsize(),
                    self.WS_queue[2].qsize(),
                    self.WS_queue[3].qsize(),
                    self.WS_queue[4].qsize(),
                    self.WS_queue[5].qsize(),
                    self.WS_queue[6].qsize(),                
                    x_new[0], 
                    x_new[1], 
                    x_new[2], 
                    x_new[3],
                    x_new[4],
                    x_new[5],
                    x_new[6],
                    x_new[7],
                    x_new[8]])
                    
    
    self.state_comp = np.concatenate((self.inventory[0:19],self.requirement,self.inv),axis=None)

    return  self.state_comp, self.demand_fulfilled

  def reward (self,a,state_comp,state_comp_old):

    #Action 0,1,2
    if(a[0] == 1 or a[1] == 1 or a[2] == 1):

      if (state_comp[0] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 3,4,5
    if(a[3] == 1 or a[4] == 1 or a[5] == 1):

      if (state_comp[1] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 6,7,8
    if(a[6] == 1 or a[7] == 1 or a[8] == 1):

      if (state_comp[2] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 9,10,11
    if(a[9] == 1 or a[10] == 1 or a[11] == 1):

      if (state_comp[3] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 12
    if(a[12] == 1):

      if (state_comp[4] > 0 or state_comp[5] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 13
    if(a[13] == 1):

      if (state_comp[4] > 0):
        self.Reward +=4
      elif(state_comp[4] == 0 and state_comp[6] > 0 and state_comp[8] == 0):
        self.Reward +=4
      elif(state_comp[4] == 0 and state_comp[6] > 0 and state_comp[8] > 0):
        self.Reward -=3
      else:
        self.Reward -= 3

    #Action 14
    if(a[14] == 1):

      if (state_comp[7] > 0):
        self.Reward +=4
      else:
        self.Reward -= 3
    
    #Action 15
    if(a[15] == 1):

      if (state_comp[8] > 0):
        self.Reward +=4
      elif(state_comp[8] == 0 and state_comp[6] > 0 and state_comp[6] == 0):
        self.Reward +=4
      elif(state_comp[8] == 0 and state_comp[6] > 0 and state_comp[6] > 0):
        self.Reward -=3
      else:
        self.Reward -= 3


    
    requirement = state_comp[0] + state_comp[1] + state_comp[2] + state_comp[3] + state_comp[4] + state_comp[5] + state_comp[6] + state_comp[7] + state_comp[8]
    #Action 9
    if(a==[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
      if (requirement == 0):
        self.Reward += 30
      
      else:
        self.Reward -= 30
    
    #Penalizing inventory accumulation
    tot_inv = np.sum(self.inventory)
    
    req = [0 for i in range(10)]
    for i in range(10):
      if (requirement[i] > 0):
        req[i] = 1
    

    time_penalty = [0 for i in range(7)]
    time_penalty[0] = self.ws_queue_copy[0].qsize()*4*((a[0]+a[1]+a[2]) * req[0]) 
    time_penalty[1] = self.ws_queue_copy[1].qsize()*3*((a[3]+a[4]+a[5]) * req[1]) 
    time_penalty[2] = self.ws_queue_copy[2].qsize()*4*((a[6]+a[7]+a[8]) * req[2])
    time_penalty[3] = self.ws_queue_copy[3].qsize()*2*((a[9]+a[10]+a[11]) * req[3])
    time_penalty[4] = self.ws_queue_copy[4].qsize()*2*((a[12]*max(req[4],req[5]))+(a[13]*max(req[5],req[6])))
    time_penalty[5] = self.ws_queue_copy[5].qsize()*3*(a[14] * req[7])
    time_penalty[6] = self.ws_queue_copy[6].qsize()*4*(a[15] * max(req[6],req[8]))


    w = [-0.5,-1]
    
    self.Reward = (w[0] * tot_inv) + (w[1]* np.sum(time_penalty))

    return self.Reward


  def step(self, a):
    
    self.time += 1

    #assert self.action_space.contains(a)
    if self.time==0:
        self.action_ = a

    if a == 0:
      action = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif a == 1:
      action = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif a == 2:
      action = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    elif a == 3:
      action = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    elif a == 4:
      action = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    elif a == 5:
      action = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    elif a == 6:
      action = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif a == 7:
      action = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    elif a == 8:
      action = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]  
    elif a == 9:
      action = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    elif a == 10:
      action = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    elif a == 11:
      action = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif a == 12:
      action = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    elif a == 13:
      action = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    elif a == 14:
      action = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    elif a == 15:
      action = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    elif a == 16:
      action = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    

    obs_t = self.state_comp

    Demand = self.demand()

    obs_t1, self.demand_fulfilled = self.transition(self.state_comp, action, self.inventory)

    self.state_comp_ = obs_t1

    reward = self.reward(action,obs_t1,self.state_comp_old)
    self.r = reward

    done = False
    

    #Terminate upon training for 1000 steps
    if (self.time ==5000):
      done = True
   
    
    truncated = False
    return obs_t1, reward, done, {}
  
  def render(self):
    #Implement viz
    pass

  def reset(self):
      
    self.processing_time = [4, 4, 4, 2, 2, 3, 2]    
    self.demand_ = np.array([0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08])
    self.arrivals = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
    self.WS_queue = [Queue() for i in range(7)]
    self.WS_queue_copy = self.WS_queue
    self.WS_put = ''
    self.WS_get = [['',''] for i in range(7)]
    self.WS_inprocess = [['',''] for i in range(7)]
    self.timestep = [[0,0] for i in range(7)]
    self.WS_occupied = [[False,False] for i in range(7)]
    self.tot_demand = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])
    self.time = 0
    self.Reward = 0
    self.products = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]

    #Initialize state components: inventory and total demand

    self.inventory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2,  6,  5, 2, 1, 0,  0,  0,  0, 0, 0, 0, 0, 0]

    self.requirement = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])

    self.state_comp = np.concatenate((self.inventory[0:19],self.requirement,self.inv),axis=None)
    self.state_comp_old = self.state_comp
    self.action_ = 0


    return self.state_comp


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward >= self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True
        

def make_env(rank, seed=0):
   
    def _init():
        env = InventoryEnv()
        env.seed(seed + rank)
        
        env = Monitor(env, filename="TRPOSetup1-8/")
        return env
         
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

  # Create the callback: check every 1000 steps
  savebest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir="TRPOSetup1-8/")
  # Save a checkpoint every 10000 steps
  checkpoint_callback = CheckpointCallback(save_freq=100, save_path="TRPOSetup1-8/")

  # Create the callback list
  callback = CallbackList([savebest_callback]) 
    
  num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
  env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
  

  #model = TRPO('MlpPolicy', env, verbose=1).learn(total_timesteps=50000000,log_interval=50)

  # save the agent
  #model.save("TRPOSetup1")
  
  model = TRPO.load("/home/weerasekara.s/Setup1/TRPO/TRPOSetup1-8.zip", env)
  model.learn(total_timesteps=50000000,log_interval=50)
  model.save("TRPOSetup1-9")
    





    
    
    