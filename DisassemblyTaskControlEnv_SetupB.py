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
    #action = 0 corresponds to [1,0,0,0,0,0,0,0,0]
    #action = 1 corresponds to [0,1,0,0,0,0,0,0,0]
    #action = 2 corresponds to [0,0,1,0,0,0,0,0,0]
    #action = 3 corresponds to [0,0,0,1,0,0,0,0,0]
    #action = 4 corresponds to [0,0,0,0,1,0,0,0,0]
    #action = 5 corresponds to [0,0,0,0,0,1,0,0,0]
    #action = 6 corresponds to [0,0,0,0,0,0,1,0,0]
    #action = 7 corresponds to [0,0,0,0,0,0,0,1,0]
    #action = 8 corresponds to [0,0,0,0,0,0,0,0,1]
    #action = 9 corresponds to [0,0,0,0,0,0,0,0,0]
    self.action_space = gym.spaces.Discrete(10)
    
    #Observation is the inventory and demand position at any given time
    self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) , high = np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]), dtype=np.int64)

    # intialize
    self.reset()

    self.seed(10)

  def seed(self,seed=None):
    if seed != None:
      np.random.seed(seed=int(seed))

  
  def demand(self):

    return [np.random.poisson(self.demand_[0]), np.random.poisson(self.demand_[1]), np.random.poisson(self.demand_[2]), np.random.poisson(self.demand_[3]), np.random.poisson(self.demand_[4]) , np.random.poisson(self.demand_[5])]
  
  def transition(self, s, a, x):
    
    self.state_comp_old = self.state_comp

    #Generate new demand for components
    Demand = self.demand()

    #Update the total demand
    for i in range(6):
      self.tot_demand[i] += Demand[i]
    
    #New product arrivals
    arr = np.array([np.random.poisson(self.arrivals[0]) , np.random.poisson(self.arrivals[1]) , np.random.poisson(self.arrivals[2]), 
                    np.random.poisson(self.arrivals[3]), np.random.poisson(self.arrivals[4]), np.random.poisson(self.arrivals[5]) ,
                    np.random.poisson(self.arrivals[6]), np.random.poisson(self.arrivals[7]), np.random.poisson(self.arrivals[8])])

    #Order - EMOLDQ
    
    #Get the amount of O in-process
    count_O = list(self.WS_queue[1].queue).count([0,1,0])
    
    occ = [0,0,0,0,0,0]
    
    if (self.WS_occupied[0] == True):
        occ[0] += 1
    if (self.WS_occupied[1] == True):
        occ[1] += 1
    if ((self.WS_occupied[1] == True) and (self.WS_inprocess[1]==[0,1,0])):
        occ[2] += 1
    if (self.WS_occupied[2] == True):
        occ[3] += 1
    if (self.WS_occupied[3] == True):
        occ[4] += 1
        occ[5] += 1
    if (self.WS_occupied[4] == True):
        occ[5] += 1 
        occ[2] += 1
    
    self.requirement[0] = max(self.tot_demand[0]-self.WS_queue[0].qsize()-x[11]-occ[0],0) #E 
    self.requirement[1] = max(self.tot_demand[1]-self.WS_queue[1].qsize()-x[12]-occ[1],0) #M 
    self.requirement[2] = max(self.tot_demand[2]-count_O-self.WS_queue[4].qsize()-x[13]-occ[2],0) #O
    self.requirement[3] = max(self.tot_demand[3]-self.WS_queue[2].qsize()-x[14]-occ[3],0) #L
    self.requirement[4] = max(self.tot_demand[4]-self.WS_queue[3].qsize()-x[15]-occ[4],0) #D
    self.requirement[5] = max(self.tot_demand[5]-self.WS_queue[3].qsize()-self.WS_queue[4].qsize()-x[16]-occ[5],0) #Q

    
    self.WS_queue_copy = self.WS_queue

    ###########Workstation Anatomy
    #WS1 - 4mins , WS2 - 6mins, WS3 - 4mins, WS4 - 2mins, WS5 - 2mins
    release = [[0,0,0] for i in range(5)]

    for i in range(5):
      #Check if any current job will be ended in this timestep, if so, release
      #the current job and add new job or set the machine free.If not, if the machine is occupied, count time
      if (self.timestep[i] == self.processing_time[i]):
        
        release[i] = self.WS_inprocess[i]
        self.WS_inprocess[i] = ''
        self.WS_occupied[i] = False
        self.timestep[i] = 0

        if (self.WS_queue[i].empty() == False):
          self.WS_inprocess[i] = self.WS_queue[i].get()
          self.WS_occupied[i] = True
          self.timestep[i] = 1

      elif (self.WS_occupied[i] == True):
        self.timestep[i] += 1
  
    actions = [[1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]]
    
    WS_index = 0
    index_start = 0
    k=0
    repeat = [3,3,1,1,1]
    index_end = [3,6,7,8,9]

    for i in repeat:

      for j in range(i):
      
        if (a == actions[k]):
          #If the machine is occupied, add the incoming orders to 
          #the queue. If not, send one from the queue for disassembly, add any that is coming in to the queue
          if (self.WS_occupied[WS_index] == True):
            self.WS_queue[WS_index].put(a[index_start:index_end[WS_index]])
          else:
            #If the queue is empty, send the incoming one to disassembly
            self.WS_inprocess[WS_index] = a[index_start:index_end[WS_index]]
            self.WS_occupied[WS_index] = True
            self.timestep[WS_index] += 1
        k += 1        
      
      
      index_start = index_end[WS_index]
      WS_index += 1

  

    #########Fulfil the demand
    x_new = np.array([max(x[11]+sum(release[0])-self.tot_demand[0], 0),
                      max(x[12]+sum(release[1])-self.tot_demand[1], 0),
                      max(x[13]+release[1][1]+sum(release[4])-self.tot_demand[2], 0),
                      max(x[14]+sum(release[2])-self.tot_demand[3], 0),
                      max(x[15]+sum(release[3])-self.tot_demand[4], 0),
                      max(x[16]+sum(release[3])+sum(release[4])-self.tot_demand[5], 0)])

    self.demand_fulfilled = np.array([min(self.tot_demand[0],x[11]+sum(release[0])), 
                                      min(self.tot_demand[1],x[12]+sum(release[1])),
                                      min(self.tot_demand[2],x[13]+release[1][1]+sum(release[4])), 
                                      min(self.tot_demand[3],x[14]+sum(release[2])),
                                      min(self.tot_demand[4],x[15]+sum(release[3])),
                                      min(self.tot_demand[5],x[16]+sum(release[3])+sum(release[4]))])
   
    for i in range (6):
      self.tot_demand[i] -= self.demand_fulfilled[i]

    #state_comp = [MOQ, MO, MLQD, LQD, QD, OQ, WIP1, WIP2, WIP3, WIP4, WIP5, E, M, O, L, D, Q, ReE, ReM, ReO, ReL, ReD, ReQ]
    ###########Update the state composition
    for i in range (6):
      if (x[i]>0):
        x[i] = x[i]-a[i+3]
      else:
        self.products[i+3] -=a[i+3]

      self.products[i+3] += arr[i+3]
    
    for i in range (3):
      self.products[i] = self.products[i] - a[i] + arr[i] 


    self.inventory = np.array([
                    max(x[0]+release[0][0],0),
                    max(x[1]+release[0][1],0),
                    max(x[2]+release[0][2],0),
                    max(x[3]+release[1][2],0),
                    max(x[4]+release[2][0],0),
                    max(x[5]+release[1][0],0),
                    self.WS_queue[0].qsize(),
                    self.WS_queue[1].qsize(),
                    self.WS_queue[2].qsize(),
                    self.WS_queue[3].qsize(),
                    self.WS_queue[4].qsize(),
                    x_new[0], 
                    x_new[1], 
                    x_new[2], 
                    x_new[3],
                    x_new[4],
                    x_new[5]])
                    
    
    self.state_comp = np.concatenate((self.inventory[0:10],(self.requirement,self.inv)),axis=None)

    return  self.state_comp, self.demand_fulfilled

  def reward (self,a,state_comp,state_comp_old):

     #Action 0,1,2
    if(a[0] == 1 or a[1] == 1 or a[2] == 1):

      if (state_comp[0] > 0):
        self.Reward +=1
      else:
        self.Reward -= 1
    
    #Action 3,4
    if(a[3] == 1 or a[4] == 1):

      if(state_comp[1] > 0 and state_comp[2] > 0):
        self.Reward += 1
      elif(state_comp[1] > 0 and state_comp[2] == 0):
        self.Reward += 1
      elif(state_comp[1] == 0 and state_comp[2] > 0 and state_comp[5] == 0):
        self.Reward += 1
      elif(state_comp[1] == 0 and state_comp[2] > 0 and state_comp[5] > 0):
        self.Reward -= 1
      elif(state_comp[1] == 0 and state_comp[2] == 0):
        self.Reward -= 1
    
    #Action 5

    if(a[5] == 1):
      if (state_comp[1] > 0):
        self.Reward += 1

      else:
        self.Reward -= 1


    #Action 6
    if(a[6] == 1):
      if (state_comp[3] > 0):
        self.Reward += 1
      else:
        self.Reward -= 1

    
    #Action 7 
    if(a[7] == 1):
      if (state_comp[4] > 0 and state_comp[5] > 0):
        self.Reward += 1

      elif(state_comp[4] > 0 and state_comp[5] == 0):
        self.Reward += 1

      elif(state_comp[4] == 0 and state_comp[5] > 0 and state_comp[2] == 0):
        self.Reward += 1

      elif(state_comp[4] == 0 and state_comp[5] > 0 and state_comp[2] > 0):
        self.Reward -= 1

      elif(state_comp[4] == 0 and state_comp[5] == 0):
        self.Reward -= 1
      

    #Action 8
    if(a[8] == 1):
      if (state_comp[5] > 0 and state_comp[2] > 0):
        self.Reward += 1

      elif(state_comp[5] > 0 and state_comp[2] == 0 and state_comp[4] == 0):
        self.Reward += 1
      
      elif(state_comp[5] > 0 and state_comp[2] == 0 and state_comp[4] > 0):
        self.Reward -= 1

      elif(state_comp[5] == 0 and state_comp[2] > 0 and state_comp[1] == 0):
        self.Reward += 1
      
      elif(state_comp[5] == 0 and state_comp[2] > 0 and state_comp[1] > 0):
        self.Reward -= 1

      elif(state_comp[5] == 0 and state_comp[2] == 0):
        self.Reward -= 1

    requirement = state_comp[0] + state_comp[1] + state_comp[2] + state_comp[3] + state_comp[4] + state_comp[5]
    #Action 9
    if(a==[0,0,0,0,0,0,0,0,0]):
      if (requirement == 0):
        self.Reward += 20
      
      else:
        self.Reward -= 20
    

    #Penalizing inventory accumulation and WS delay
    tot_inv = np.sum(self.inventory)

    req = [0,0,0,0,0,0]
    if (requirement[0] > 0):
      req[0] = 1
    if (requirement[1] > 0):
      req[1] = 1
    if (requirement[2] > 0):
      req[2] = 1
    if (requirement[3] > 0):
      req[3] = 1
    if (requirement[4] > 0):
      req[4] = 1
    if (requirement[5] > 0):
      req[4] = 1

    time_penalty = [0,0,0,0,0]
    time_penalty[0] = self.ws_queue_copy[0].qsize()*4*((a[0]+a[1]+a[2]) * req[0]) 
    time_penalty[1] = self.ws_queue_copy[1].qsize()*3*((a[3]+a[4]+a[5])*req[1] + a[4]*max(req[1],req[2]))
    time_penalty[2] = 0
    time_penalty[3] = self.ws_queue_copy[3].qsize()*2*a[7]*(max(req[4],req[5]))
    time_penalty[4] = self.ws_queue_copy[4].qsize()*2*a[8]*(max(req[2],req[5]))


    w = [-1,-0.7]
    
    self.Reward = (w[0] * tot_inv) + (w[1]* np.sum(time_penalty))

    return self.Reward


  def step(self, a):
    
    self.time += 1

    #assert self.action_space.contains(a)
    if self.time==0:
        self.action_ = a

    if a == 0:
      action = [1,0,0,0,0,0,0,0,0]
    elif a == 1:
      action = [0,1,0,0,0,0,0,0,0]
    elif a == 2:
      action = [0,0,1,0,0,0,0,0,0]
    elif a == 3:
      action = [0,0,0,1,0,0,0,0,0]
    elif a == 4:
      action = [0,0,0,0,1,0,0,0,0]
    elif a == 5:
      action = [0,0,0,0,0,1,0,0,0]
    elif a == 6:
      action = [0,0,0,0,0,0,1,0,0]
    elif a == 7:
      action = [0,0,0,0,0,0,0,1,0]
    elif a == 8:
      action = [0,0,0,0,0,0,0,0,1]  
    elif a == 9:
      action = [0,0,0,0,0,0,0,0,0]
    
    obs_t = self.state_comp

    Demand = self.demand()

    obs_t1, self.demand_fulfilled = self.transition(self.state_comp, action, self.inventory)

    self.state_comp_ = obs_t1

    reward = self.reward(action,obs_t1,self.state_comp_old)
    self.r = reward

    done = False
    

    #Terminate upon training for 1000 steps
    if (self.time ==1000):
      done = True
   
    
    truncated = False
    return obs_t1, reward, done, {}
  
  def render(self):
    #Implement viz
    pass

  def reset(self):
      
    self.processing_time = [4, 6, 4, 2, 2]    
    self.demand_ = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
    self.arrivals = np.array([3,3,3,3,3,3,3,3,3])
    self.WS_queue = [Queue() for i in range(5)]
    self.WS_queue_copy = self.WS_queue
    self.WS_put = ''
    self.WS_get = ['' for i in range(5)]
    self.WS_inprocess = ['' for i in range(5)]
    self.timestep = [0 for i in range(5)]
    self.WS_occupied = [False for i in range(5)]
    self.tot_demand = np.array([30, 30, 30, 30, 30, 30])
    self.time = 0
    self.Reward = 0
    self.products = [1000,1000,1000,1000,1000,1000,1000,1000,1000]

    #Initialize state components: inventory and total demand
    #state_comp = [ReE, ReM, ReO, ReL, ReD, ReQ]
    self.inventory = [0, 0, 0, 0, 0, 0, 2, 1, 2,  6,  5,  0,  0,  0,  0, 0, 0]
    self.inv = np.array([0, 0, 0, 0, 0])

    self.requirement = np.array([30,30,30,30,30,30])

    self.state_comp = np.concatenate((self.inventory[0:10],(self.requirement,self.inv)),axis=None)
    self.state_comp_old = self.state_comp
    self.action_ = 0


    return self.state_comp






    
    
    