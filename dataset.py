from torch.utils.data import Dataset
import torch
import pandas as pd

class MyDataset(Dataset):
    
  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)

    x=price_df.iloc[:,0:54].values
    y=price_df.iloc[:,54].values-1

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]