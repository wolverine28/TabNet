# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax

class FeatureTransformer(nn.Module):
    def __init__(self, input_size, n_d, n_a, shared_block, independent_step, vitrual_batch_size):
        super(FeatureTransformer, self).__init__()
        self.input_size = input_size
        self.n_d = n_d
        self.n_a = n_a
        self.output_size = self.n_d+ self.n_a
        self.vitrual_batch_size = vitrual_batch_size

        self.shared_glu_layers = shared_block
        self.independent_glu_layers = Blocks(self.output_size, n_d, n_a, independent_step, vitrual_batch_size)

    def forward(self, x):
        x = self.shared_glu_layers(x)
        x = self.independent_glu_layers(x)
        d,a = x[:,:self.n_d], x[:,self.n_d:]
        return d, a

class FC_BN_GLU(nn.Module):
    def __init__(self, input_size, output_size, vitrual_batch_size):
        super(FC_BN_GLU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Linear(input_size, output_size*2, bias=False)
        # self.bn = nn.BatchNorm1d(output_size*2)
        self.bn = GBN(output_size*2,vbs=vitrual_batch_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_size], torch.sigmoid(x[:, self.output_size :]))
        return out

class Blocks(nn.Module):
    def __init__(self, input_size, n_d, n_a, step, vitrual_batch_size):
        super(Blocks, self).__init__()
        self.input_size = input_size
        self.n_d = n_d
        self.n_a = n_a
        self.step = step
        self.output_size = self.n_d+ self.n_a
        self.sqrt_half = torch.sqrt(torch.tensor(0.5))
        self.vitrual_batch_size = vitrual_batch_size
        
        self.glu_layers = nn.ModuleList()
        self.glu_layers.append(FC_BN_GLU(self.input_size, self.output_size, self.vitrual_batch_size))
        for i in range(1,self.step):
            self.glu_layers.append(FC_BN_GLU(self.output_size, self.output_size, self.vitrual_batch_size))

    def forward(self, x):
        x = self.glu_layers[0](x)
        for i in range(1,self.step):
            x = (self.glu_layers[i](x)+x)*self.sqrt_half
        return x
    
class AttentiveTransformer(nn.Module):
    def __init__(self, n_a, input_size, vitrual_batch_size):
        super(AttentiveTransformer, self).__init__()
        self.n_a = n_a
        self.input_size = input_size
        self.vitrual_batch_size = vitrual_batch_size
        
        self.fc = nn.Linear(n_a, input_size, bias=False)
        # self.bn = nn.BatchNorm1d(input_size)
        self.bn = GBN(input_size,vbs=vitrual_batch_size)
        self.sparsemax = Sparsemax(dim=1)


    def forward(self, x, prior_sclae):
        x = self.fc(x)
        x = self.bn(x)
        x =  self.sparsemax(x*prior_sclae)
        return x
    
class TabNet(nn.Module):
    def __init__(self, input_size, output_size, n_d, n_a, shared_step, independent_step, global_step, batch_size, vitrual_batch_size, gamma, epsilon, device):
        super(TabNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_d = n_d
        self.n_a = n_a
        self.shared_step = shared_step
        self.independent_step = independent_step
        self.global_step = global_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.vitrual_batch_size = vitrual_batch_size
        self.device = device
        
        self.bn = nn.BatchNorm1d(input_size)
        # self.bn = GBN(input_size)
        self.shared_block = Blocks(self.input_size, self.n_d, self.n_a, self.shared_step, self.vitrual_batch_size)
        
        self.FTsteps = nn.ModuleList()
        self.FTsteps.append(FeatureTransformer(input_size=self.input_size, n_d=self.n_d, n_a=self.n_a, shared_block=self.shared_block, independent_step=self.independent_step, vitrual_batch_size=self.vitrual_batch_size))
        for _ in range(self.global_step):
            self.FTsteps.append(FeatureTransformer(input_size=self.input_size, n_d=self.n_d, n_a=self.n_a, shared_block=self.shared_block, independent_step=self.independent_step, vitrual_batch_size=self.vitrual_batch_size))
        
        self.ATsteps = nn.ModuleList()
        for _ in range(self.global_step):
            self.ATsteps.append(AttentiveTransformer(n_a=self.n_a, input_size=self.input_size, vitrual_batch_size=self.vitrual_batch_size))
        
        # if prior_scale is not None:    
        #     self.prior_scale = prior_scale.to(device)
        # else:
        #     self.prior_scale = torch.ones((self.batch_size, self.input_size),device=device)
            
        # self.Mlist = []
        # self.dlist = []
        self.relu = nn.ReLU()
        self.outLayer = nn.Linear(self.n_d, self.output_size)

    def L_sparse(self, Mlist):
        return (-Mlist*torch.log(Mlist+self.epsilon)).sum()/(self.global_step*self.batch_size)   

    def forward(self, x):
        Mlist = []
        dlist = []
        prior_scale = torch.ones((x.size(0), self.input_size),device=self.device)
            
        bn_x = self.bn(x)
        _, a = self.FTsteps[0](bn_x)
        for s in range(self.global_step):
            M = self.ATsteps[s](a, prior_scale)
            Mlist.append(M)
            prior_scale = torch.stack([self.gamma-m for m in Mlist]).prod(dim=0)
            d,a = self.FTsteps[s+1](bn_x*M)
            dlist.append(self.relu(d))
            
        dout = torch.stack(dlist).sum(dim=0)
        out = self.outLayer(dout)
        return out, torch.stack(Mlist), torch.stack(dlist)

class GBN(nn.Module):
    def __init__(self,inp,vbs=128,momentum=0.7):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp,momentum=momentum)
        self.vbs = vbs
    def forward(self,x):
        chunk = torch.chunk(x,x.size(0)//self.vbs,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res,0)