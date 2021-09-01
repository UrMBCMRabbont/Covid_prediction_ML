import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from MyModule import MyModel
from MyDataset import MyDataset

lr = 0.001
batch_size = 10
epochs = 30
device = "cpu"

result = pd.read_csv("covid.train.csv")
keys = list(result.columns.values)

    
#turn into tensor
a = torch.FloatTensor(result.iloc[0][0:82])
b = torch.FloatTensor(result.iloc[0][82:])
result = pd.DataFrame.to_numpy(result)
result = torch.FloatTensor(result)

#standardize
temp = result[:,0:40]
temp2 = result[:,40:-1]
temp3 = result[:,-1:]
temp2 = (temp2 - torch.min(temp2,0)[0]) / (torch.max(temp2,0)[0] - torch.min(temp2,0)[0])

#remove the items with low correlation
new_data = []
new_keys = []
show_corre = 0
for i in range(temp2.shape[1]):
    
    show_corre = np.corrcoef(temp2[:,i],temp3.squeeze())
    correlation = np.corrcoef(temp2[:,i],temp3.squeeze())[0][1]
    if correlation > 0.35 or correlation < -0.35:
        new_keys.append(keys[i+40])
        temp = torch.cat((temp,torch.unsqueeze(temp2[:,i],1)),1)
        
temp = torch.cat((temp,temp3),1)
result = temp



dataset = MyDataset(result)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
model = MyModel().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr)

loss_con = []
epoch_number = 0
trigger = 0
trigger_2 = 0
average_loss_list = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for x,y in dataloader: #attention
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)


        loss = criterion(pred, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print(epoch_loss)
    epoch_number = epoch_number + 1
    
    if (epoch_loss < 70000) and (trigger == 0):
        trigger = 1
        loss_con.append(epoch_loss)
    elif (trigger == 1):
        loss_con.append(epoch_loss)
    elif (epoch_loss > 70000) and (trigger == 0):
        pass


model.eval()
total_loss = 0
pred_results = []
final_loss = []

for x,y in dataloader:
    x = x.to(device)
    epoch_final_loss = 0
    
    with torch.no_grad():
        predict = model(x)
    pred_results.append(predict.cpu())

    loss_final = criterion(predict, y)
    epoch_final_loss += loss_final.item()

    print(epoch_final_loss)
    final_loss.append(epoch_final_loss)

print(pred_results)