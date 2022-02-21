#%%
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchmetrics

from networks import TabNet
import numpy as np
from torchviz import make_dot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# sns.set_theme(style="ticks")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
input_size = 9
output_size = 1
n_d = 8
n_a = 8
shared_step = 2
independent_step = 2
global_step = 5
gamma = 1.5
epsilon = 0.001
BATCH_SIZE = 3000
vitrual_batch_size = 100
lam_sparse = 0.005

class MyDataset(Dataset):
    
  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)

    x=price_df.iloc[:,:9].values
    y=price_df.iloc[:,9].values

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

dataset = MyDataset('./syn5.csv')
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
test_dataloader = DataLoader(val_set, batch_size=len(dataset)-int(len(dataset)*0.8), shuffle=True,drop_last=True)


TN = TabNet(input_size,output_size, n_d, n_a, shared_step, independent_step, global_step, BATCH_SIZE, vitrual_batch_size, gamma, epsilon, device).to(device)
optimizer = optim.Adam(TN.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
criterion = nn.BCEWithLogitsLoss()


def feat_imp(Mlist,dlist):
    eta = dlist.sum(dim=2)
    imp = torch.zeros_like(Mlist[0,:])
    for n in range(5):
        imp +=Mlist[n,:]*eta[n,:].view(3000,1)
    imp = imp/imp.sum(1).view(-1,1)
    return torch.tensor(imp).cpu().numpy()


# %%
for it in range(4000):
    for i, (x,y) in enumerate(dataloader):
        TN.train()
        count = len(dataloader)*it+i
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output, Mlist, dlist = TN(x)
        loss = criterion(output.view(-1),y) + lam_sparse*TN.L_sparse(Mlist)
        loss.backward()
        optimizer.step()

        pred = (1*(torch.sigmoid(output.view(-1))>0.5))
        BA = torchmetrics.functional.accuracy(pred,y.type(torch.int32),average='macro',num_classes=2).item()
        # BA = torchmetrics.functional.accuracy(pred,y.type(torch.int32)).item()

        # feat_imp = feat_imp(Mlist,dlist)
        # sns.heatmap(feat_imp[np.argsort(feat_imp[:,4]),:], vmin=0, vmax=1)
        # plt.show()
        
        if count % 200 == 1:
            scheduler.step()
            torch.save(TN.state_dict(), './model/TabNet_synthetic_'+str(count)+'.pth')
            
        if count % 100 == 0:
            for i, (x,y) in enumerate(test_dataloader):
                TN.eval()
                x = x.to(device)
                y = y.to(device)
                
                output, Mlist, dlist = TN(x)
            
                pred = (1*(torch.sigmoid(output.view(-1))>0.5))
                BA_test = torchmetrics.functional.accuracy(pred,y.type(torch.int32),average='macro',num_classes=2).item()
                
                
            print('iter: {:d}, loss: {:.3f}, acc: {:.3f} test_acc: {:.3f}'.format(count, loss.item(), BA, BA_test))