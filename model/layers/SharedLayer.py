# -*- coding: utf-8 -*-
"""
   File Name：     SharedLayer.py
   Description : 
   Author :       zhaowei
   date：          2025/9/3
"""
__author__ = 'zhaowei'
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedLayer(nn.Module):
    def __init__(self, d_kv, sz_c, dropout, h_dim, temp):
        super(SharedLayer, self).__init__()
        self.sz_c = sz_c
        self.d_kv = d_kv
        self.temp = temp
        self.MLP = nn.Sequential(
            nn.Linear(d_kv, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, d_kv)
        )
        self.fc1 = nn.Linear(32*32,32)
    def forward(self,x):
        batchsize = x.size(0)
        Z = []
        for i in range(self.sz_c):
            xi = x[:,i,:,:].contiguous().view(-1,self.d_kv)
            xi = self.MLP(xi) 
            xi = xi.view(batchsize,self.d_kv,-1).view(batchsize,-1)         #(batchsize*d_kv,d_kv) to (batchsize,d_kv,d_kv) to (batchsize,d_kv*d_kv)
            xi = self.fc1(xi)                                               #(batchsize,d_kv)
            Z.append(xi)
        Z = torch.stack(Z,dim=0).permute(1,0,2)       #(sz_c,batchsize,d_kv) to (batchsize,sz_c,d_kv)
             
        # neg_sample_num = batchsize-1
        if batchsize<=64:
            neg_sample_num = 31
        else:
            neg_sample_num = batchsize//4
        neg_sim = []
        for i in range(batchsize):    
            indices = torch.cat([torch.arange(i), torch.arange(i+1, batchsize)])
            sampled_data = Z[indices[torch.randint(0, len(indices), (neg_sample_num,))]]        #(neg_sample_num,sz_c,d_kv)
            neg_sim.append(torch.mean(F.cosine_similarity(Z[i],sampled_data,dim=2),dim=1))    
        neg_sim = torch.stack(neg_sim,dim=0)                                        # neg_sim  (batchsize,neg_sample_num)
        total_similarity = 0
        num_combinations = 0  
        for j in range(self.sz_c):
            for k in range(j + 1, self.sz_c):
                cos_sim = F.cosine_similarity(Z[:, j, :], Z[:, k, :], dim=1)
                total_similarity += cos_sim
                num_combinations += 1
        pos_sim = total_similarity / num_combinations       # pos_sim    (batchsize,)
        self.l = torch.exp(pos_sim/self.temp)/torch.sum(torch.exp(neg_sim/self.temp),dim=1)

        Z = torch.mean(Z,dim=1).squeeze(1)            #(batchsize,d_kv)
        return Z

    def Loss_MLP(l):
        # Loss = -torch.log(torch.mean(l))
        Loss = -torch.mean(torch.log(l))
        return Loss

