import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self, file):
        self.data = file

    def __getitem__(self,index): 
        
        train_data = self.data[index,0:56]
#         print(train_data.shape)
        train_target = self.data[index,-1:]
#         print(train_target)
        return train_data, train_target
    
    
    def __len__(self):
        return len(self.data)
