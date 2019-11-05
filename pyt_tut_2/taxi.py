import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
 

np.set_printoptions(threshold=sys.maxsize)
env = gym.make("Taxi-v3")

state_space = env.observation_space.n
action_space = env.action_space.n
 
qtable = np.zeros((state_space, action_space))

epsilon = 1.0           #Greed 100%
 
epsilon_min = 0.005     #Minimum greed 0.05%
 
epsilon_decay = 0.99993 #Decay multiplied with epsilon after each episode
 
episodes = 50000        #Amount of games
 
max_steps = 100         #Maximum steps per episode
 
learning_rate = 0.65
 
gamma = 0.65

print(qtable)