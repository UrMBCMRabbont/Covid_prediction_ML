import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(56, 32),
            nn.ReLU(),
            nn.Linear(32,16),
#             nn.ReLU(),
#             nn.Linear(16,1)
            
        )
    def forward(self, x):
        return self.net(x).squeeze()
