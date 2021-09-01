import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import csv

path = "covid.train.csv"
with open(path,'r') as fp:
    csv_file = csv.reader((fp))
    print(type(csv_file))
    csv_file = list(csv_file)
    
result = pd.read_csv("covid.train.csv")

result.drop("id", inplace=True, axis=1)
result.drop("wearing_mask", inplace=True, axis=1)
result.drop("wearing_mask.2", inplace=True, axis=1)
result.drop("wearing_mask.1", inplace=True, axis=1)
result.drop("felt_isolated", inplace=True, axis=1)
result.drop("felt_isolated.2", inplace=True, axis=1)
result.drop("felt_isolated.1", inplace=True, axis=1)
result.drop("depressed", inplace=True, axis=1)
result.drop("depressed.2", inplace=True, axis=1)
result.drop("depressed.1", inplace=True, axis=1)
result.drop("anxious", inplace=True, axis=1)
result.drop("anxious.2", inplace=True, axis=1)
result.drop("anxious.1", inplace=True, axis=1)
print(result)



