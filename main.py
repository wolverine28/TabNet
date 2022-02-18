#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics

from networks import TabNet
import numpy as np
from torchviz import make_dot
# import seaborn as sns
# import pandas as pd
# sns.set_theme(style="ticks")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# Orange Skin
N = 10000
X = np.random.normal(0,1,(N,10))
Y = np.logical_and(np.sum(X[:,range(4)]**2,1)>=9,np.sum(X[:,range(4)]**2,1)<=16)*1
# df_X = pd.DataFrame(X)
# df_Y = pd.DataFrame(Y)

# df = pd.concat([df_X,df_Y],axis=1,names=['X','Y'])
# df.columns = [0,1,2,3,4,5,6,7,8,9,'Y']
# sns.pairplot(df,hue='Y')

#%%
input_size = 10
output_size = 1
n_d = 16
n_a = 16
shared_step = 2
independent_step = 2
global_step = 5
gamma = 2
epsilon = 0.001
BATCH_SIZE = 3000
vitrual_batch_size = 100
lam_sparse = 0.02


TN = TabNet(input_size,output_size, n_d, n_a, shared_step, independent_step, global_step, BATCH_SIZE, vitrual_batch_size, gamma, epsilon, device).to(device)
optimizer = optim.Adam(TN.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
criterion = nn.BCEWithLogitsLoss()


# %%
for iter in range(10000):
    idx = np.random.randint(0,len(X),BATCH_SIZE)
    
    x = torch.tensor(X[idx],dtype=torch.float32,device=device)
    y = torch.tensor(Y[idx],dtype=torch.float32,device=device)
    
    optimizer.zero_grad()
    output, Mlist, dlist = TN(x)
    loss = criterion(output.view(-1),y) + lam_sparse*TN.L_sparse(Mlist)
    loss.backward()
    optimizer.step()

    pred = (1*(torch.sigmoid(output.view(-1))>0.5)==y)
    BA = torchmetrics.functional.accuracy(pred,y.type(torch.int32),average='macro',num_classes=2).item()

    
    if iter % 4000 == 1:
        scheduler.step()
        
    if iter % 100 == 0:
        print(loss.item(), BA)