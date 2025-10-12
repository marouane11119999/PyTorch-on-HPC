import torch.nn as nn
import torch.nn.functional as F
import torch
import logging, time, os, argparse
from pathlib import Path
import pandas as pd

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # first conv layer + relu
        x = self.conv1(x)
        x = F.relu(x) # this returns a (32, 26, 26) tensor
        
        # second conv layer + relu + maxpool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # this returns a (64, 12, 12) tensor
        
        # flatten
        x = torch.flatten(x, 1) # this returns a (9216,) tensor ready for fc layer
        x = self.fc1(x)
        x = F.relu(x) # this returns a (128,) tensor
        
        x = self.fc2(x) # this returns a (10,) tensor
        output = F.log_softmax(x, dim=1) # transform to log-probabilities
        return output
    

        
