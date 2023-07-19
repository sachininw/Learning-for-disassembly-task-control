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

import os
import argparse
import json
import math

from typing import Optional, Union

class InventoryEnv(gym.Env):
  def __init__(self, *args, **kwargs):
    #action = 0 corresponds to [1,0,0,0,0,0]
    #action = 1 corresponds to [0,1,0,0,0,0]
    #action = 2 corresponds to [0,0,1,0,0,0]
    #action = 3 corresponds to [0,0,0,1,0,0]
    #action = 4 corresponds to [0,0,0,0,1,0]
    #action = 5 corresponds to [0,0,0,0,0,1]
    #action = 6 corresponds to [0,0,0,0,0,0]
    self.action_space = gym.spaces.Discrete(7)
    
    #Observation is the inventory and demand position at any given time
    self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0]) , high = np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]), dtype=np.int64)

    # intialize
    self.reset()

    self.seed(5)

  def seed(self,seed=None):
    if seed != None:
      np.random.seed(seed=int(seed))

  
  def demand(self):

    return [np.random.poisson(self.demand_[0]), np.random.poisson(self.demand_[1]), np.random.poisson(self.demand_[2]), np.random.poisson(self.demand_[3]) , np.random.poisson(self.demand_[4])]
  
  def transition(self, s, a, x):

    #Generate new demand for components
    Demand = self.demand()

    #Update the total demand
    for i in range(5):
      self.tot_demand[i] += Demand[i]
    
    #New product arrivals
    arr = np.array([np.random.poisson(self.arrivals[0]) , np.random.poisson(self.arrivals[1]) , np.random.poisson(self.arrivals[2]), 
                    np.random.poisson(self.arrivals[3]), np.random.poisson(self.arrivals[4]), np.random.poisson(self.arrivals[5])])

    #Order - EMOLDQ
    
    #Get the amount of O in-process
    count_O = list(self.WS_queue[0].queue).count([0,1,0])
    
    occ = [0,0,0,0,0]
    
    if (self.WS_occupied[0] == True):
        occ[0] += 1
    if ((self.WS_occupied[0] == True) and (self.WS_inprocess[1]==[0,1,0])):
        occ[1] += 1
    if (self.WS_occupied[1] == True):
        occ[2] += 1
    if (self.WS_occupied[2] == True):
        occ[3] += 1
        occ[4] += 1
    if (self.WS_occupied[3] == True):
        occ[1] += 1
        occ[4] += 1
    #[LDQ, DQ, OQ, W1, W2, W3, W4, E, O, L, D, Q]
    self.requirement[0] = max(self.tot_demand[0]-self.WS_queue[0].qsize()-x[7]-occ[0],0) #E 
    self.requirement[1] = max(self.tot_demand[1]-count_O-self.WS_queue[3].qsize()-x[8]-occ[1],0) #O
    self.requirement[2] = max(self.tot_demand[2]-self.WS_queue[1].qsize()-x[9]-occ[2],0) #L
    self.requirement[3] = max(self.tot_demand[3]-self.WS_queue[2].qsize()-x[10]-occ[3],0) #D
    self.requirement[4] = max(self.tot_demand[4]-self.WS_queue[2].qsize()-self.WS_queue[3].qsize()-x[11]-occ[4],0) #Q

    
    self.WS_queue_copy = self.WS_queue

    ###########Workstation Anatomy
    #WS1 - 4mins , WS2 - 6mins, WS3 - 4mins, WS4 - 2mins, WS5 - 2mins
    release = [[0,0,0] for i in range(4)]

    for i in range(4):
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
  
    actions = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0]]
    
    WS_index = 0
    index_start = 0
    k=0
    repeat = [3,1,1,1]
    index_end = [3,4,5,6]

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
    x_new = np.array([max(x[7]+sum(release[0])-self.tot_demand[0], 0),
                      max(x[8]+release[0][1]+sum(release[3])-self.tot_demand[1], 0),
                      max(x[9]+sum(release[1])-self.tot_demand[2], 0),
                      max(x[10]+sum(release[2])-self.tot_demand[3], 0),
                      max(x[11]+sum(release[2])+sum(release[3])-self.tot_demand[4], 0)])

    self.demand_fulfilled = np.array([min(self.tot_demand[0],x[7]+sum(release[0])), 
                                      min(self.tot_demand[1],x[8]+release[0][1]+sum(release[3])), 
                                      min(self.tot_demand[2],x[9]+sum(release[1])),
                                      min(self.tot_demand[3],x[10]+sum(release[2])),
                                      min(self.tot_demand[4],x[11]+sum(release[2])+sum(release[3]))])
   
    for i in range (5):
      self.tot_demand[i] -= self.demand_fulfilled[i]

    #state_comp = [MOQ, MO, MLQD, LQD, QD, OQ, WIP1, WIP2, WIP3, WIP4, WIP5, E, M, O, L, D, Q, ReE, ReM, ReO, ReL, ReD, ReQ]
    ###########Update the state composition
    for i in range (3):
      if (x[i]>0):
        x[i] = x[i]-a[i+3]
      else:
        self.products[i+3] -=a[i+3]

      self.products[i+3] += arr[i+3]
    
    for i in range (3):
      self.products[i] = self.products[i] - a[i] + arr[i] 


    self.inventory = np.array([
                    max(x[0]+release[0][2],0),
                    max(x[1]+release[1][0],0),
                    max(x[2]+release[0][0],0),
                    self.WS_queue[0].qsize(),
                    self.WS_queue[1].qsize(),
                    self.WS_queue[2].qsize(),
                    self.WS_queue[3].qsize(),
                    x_new[0], 
                    x_new[1], 
                    x_new[2], 
                    x_new[3],
                    x_new[4]])                 
    
    self.state_comp = self.inventory

    return  self.state_comp, self.demand_fulfilled
   
  
  def reward(self,a,requirement):
    
    #Action 0,1,2
    if(a[0] == 1 or a[2] == 1):

      if (requirement[0] > 0):
        self.Reward +=2
      else:
        self.Reward -= 1
    
    if(a[1] == 1):

      if(requirement[0] > 0 and requirement[1] > 0):
        self.Reward +=2
      elif(requirement[0] > 0 and requirement[1] == 0):
        self.Reward +=2  
      elif(requirement[0] == 0 and requirement[1] > 0 and requirement[4]>0):
        self.Reward -=1     
      elif(requirement[0] == 0 and requirement[1] > 0 and requirement[4]==0):
        self.Reward +=2
    
    #Action 3
    if(a[3] == 1):
      if (requirement[2] > 0):
        self.Reward +=2
      else:
        self.Reward -= 1

    #Action 4
    if(a[4] == 1):
      if(requirement[3]>0):
        self.Reward +=2
      elif(requirement[3]==0 and requirement[4]>0 and requirement[1]>0):
        self.Reward -=1
      elif(requirement[3]==0 and requirement[4]>0 and requirement[1]==0):
        self.Reward +=2
      elif(requirement[3]==0 and requirement[4]==0):
        self.Reward -=1
    
    #Action 5

    if(a[5] == 1):
      if (requirement[1] > 0 and requirement[4] > 0):
        self.Reward += 2
      elif(requirement[1] >0 and requirement[4] == 0 and requirement[0] == 0):
        self.Reward += 2
      elif(requirement[1] >0 and requirement[4] == 0 and requirement[0] > 0):
        self.Reward -= 1
      elif(requirement[1] == 0 and requirement[4] > 0 and requirement[3] == 0):
        self.Reward += 2
      elif(requirement[1] == 0 and requirement[4] > 0 and requirement[3] > 0):
        self.Reward -= 1
      elif(requirement[1] == 0 and requirement[4] == 0):
        self.Reward -= 1
        
    #Action 6
    if(a==[0,0,0,0,0,0]):
      if (sum(requirement) == 0):
        self.Reward += 10
      
      else:
        self.Reward -= 10
    
    self.total_inventory =  np.sum(self.state_comp)

    req = [0,0,0,0,0]
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

    time_penalty = [0,0,0,0]
    time_penalty[0] = self.ws_queue_copy[0].qsize()*4*((a[0]+a[2]) * req[0] + a[1] * max(req[0],req[1]))
    time_penalty[1] = self.ws_queue_copy[1].qsize()*4*(a[3])*(req[2])
    time_penalty[2] = self.ws_queue_copy[2].qsize()*2*a[4]*(max(req[3],req[4]))
    time_penalty[3] = self.ws_queue_copy[3].qsize()*2*a[5]*(max(req[1],req[4]))


    w = [-0.8,-0.7]
    
    self.Reward = (w[0] * self.total_inventory) + (w[1]* np.sum(time_penalty))

    return self.Reward

  def step(self, a):
    
    self.time += 1

    #assert self.action_space.contains(a)
    if self.time==0:
        self.action_ = a

    if a == 0:
      action = [1,0,0,0,0,0]
    elif a == 1:
      action = [0,1,0,0,0,0]
    elif a == 2:
      action = [0,0,1,0,0,0]
    elif a == 3:
      action = [0,0,0,1,0,0]
    elif a == 4:
      action = [0,0,0,0,1,0]
    elif a == 5:
      action = [0,0,0,0,0,1]
    elif a == 6:
      action = [0,0,0,0,0,0]
    
    obs_t = self.state_comp

    Demand = self.demand()

    obs_t1, self.demand_fulfilled = self.transition(self.state_comp, action, self.inventory)

    self.state_comp_ = obs_t1

    reward = self.reward(action,obs_t1)
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
      
    self.processing_time = [4, 4, 2, 2]    
    self.demand_ = np.array([0.1,0.1,0.1,0.1,0.1])
    self.arrivals = np.array([3,3,3,3,3,3])
    self.WS_queue = [Queue() for i in range(4)]
    self.WS_queue_copy = self.WS_queue
    self.WS_put = ''
    self.WS_get = ['' for i in range(4)]
    self.WS_inprocess = ['' for i in range(4)]
    self.timestep = [0 for i in range(4)]
    self.WS_occupied = [False for i in range(4)]
    self.tot_demand = np.array([30, 30, 30, 30, 30])
    self.time = 0
    self.Reward = 0
    self.products = [1000,1000,1000,1000,1000,1000]

    #Initialize state components: inventory and total demand
    #state_comp = [ReE, ReM, ReO, ReL, ReD, ReQ]
    self.inventory = [0, 2, 0, 1, 1, 5,  0, 0,  2,  0,  1, 1]

    self.requirement = np.array([30,30,30,30,30])

    self.state_comp = self.inventory
    self.action_ = 0


    return self.state_comp






    
    
    