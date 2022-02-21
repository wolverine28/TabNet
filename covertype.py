
#%%
import enum
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics

from networks import TabNet
from dataset import MyDataset
from torch.utils.data import DataLoader

import numpy as np
from torchviz import make_dot
# import seaborn as sns
import pandas as pd
# sns.set_theme(style="ticks")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
input_size = 54
output_size = 7
n_d = 64
n_a = 64
shared_step = 2
independent_step = 2
global_step = 5
gamma = 1.5
epsilon = 0.001
BATCH_SIZE = 16384
vitrual_batch_size = 512
lam_sparse = 0.0001
#%%
# Forest Covertype data
dataset = MyDataset("D:\Datasets\Forest Cover Type\covtype.data")
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
test_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)



TN = TabNet(input_size,output_size, n_d, n_a, shared_step, independent_step, global_step, BATCH_SIZE, vitrual_batch_size, gamma, epsilon, device).to(device)
optimizer = optim.Adam(TN.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
criterion = nn.CrossEntropyLoss()
m = nn.LogSoftmax(dim=1)
# %%
for it in range(130000):
    for i, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y = y.type(torch.long)
        
        optimizer.zero_grad()
        output, Mlist, dlist = TN(x)
        loss = criterion(output,y) + lam_sparse*TN.L_sparse(Mlist)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output,1)
        BA = torchmetrics.functional.accuracy(pred,y).item()

    
        if len(dataloader)*it+i % 500 == 1:
            scheduler.step()
            torch.save(TN.state_dict(), './model/TabNet_'+str(len(dataloader)*it+i)+'.pth')
        
    if it % 1 == 0:
        all_BA = []
        all_loss = []
        for x,y in test_dataloader:
            x, y = x.to(device), y.to(device)
            y = y.type(torch.long)
            
            output, Mlist, dlist = TN(x)
            
            pred = torch.argmax(output,1)
            BA = torchmetrics.functional.accuracy(pred,y).item()
            
            all_BA.append(BA)    
        print('iter: {:d}, loss: {:.3f}, acc: {:.3f} test acc: {:.3f}'.format(len(dataloader)*it+i, loss.item(), BA, np.mean(all_BA)))