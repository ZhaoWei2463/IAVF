# -*- coding: utf-8 -*-
"""
   File Name：     IAVF.py
   Description : IAVF
   Author :       zhaowei
   date：          2025/9/3
"""
__author__ = 'zhaowei'
from .layers.MGM import MGM
from .layers.EmbeddingTransform import EmbeddingTransform
from torch import nn
import torch.nn.functional as F
from .layers.MIM import MIM
from .layers.CGIFM import CGIFM
class IAVF(nn.Module):
    def __init__(self, in_channels, out_channels, gcn_nums, dropout,
                 sz_c, graph_norm, gcn_h_dim, device, arc_type,
                 d_kv, att_drop_ratio, share_h_dim, temp, gama, gcn_dropout=0.0):
        super(IAVF, self).__init__()

        self.gama = gama
        self.share_h_dim = share_h_dim
        self.temp = temp
        self.att_drop_ratio = att_drop_ratio
        self.d_kv = d_kv
        self.arc_type = arc_type
        self.device = device
        self.gcn_h_dim = gcn_h_dim
        self.graph_norm = graph_norm
        self.sz_c = sz_c
        self.dropout = dropout

        self.MGL = MGM(in_channels, gcn_h_dim, sz_c, gcn_nums, gcn_dropout, device, arc_type, graph_norm)
        self.gcn_h_dim = gcn_h_dim // sz_c

        self.EmTran = EmbeddingTransform(self.d_kv, self.device, self.gcn_h_dim, self.att_drop_ratio)

        self.crossview = MIM(self.sz_c,self.dropout)

        self.CGIFM = CGIFM(self.d_kv, self.sz_c, self.dropout, self.share_h_dim, self.temp)

        self.fc = nn.Linear(self.d_kv, out_channels)
        self.dropout = nn.Dropout(dropout) 
        self.reset_parameters()

    def reset_parameters(self):
        all_res = [self.MGL, self.EmTran, self.fc]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    def forward(self, data, edge_index, batch):
        # Multi-channel encoder
        z = self.MGL(data, edge_index, batch)       #(sz_c,N,d_h)

        # CNN Decoder
        z = self.EmTran(z, batch)
        # Cross-view interaction
        z = self.crossview(z)        #(batchsize,c,N,F)

        # CGIFM
        z = self.CGIFM(z)
        z = self.dropout(F.relu(self.fc(z)))
        return z

    def loss(self, y_pre, y_true):
        loss1 = CGIFM.Loss_MLP(self.CGIFM.l)
        lb = F.cross_entropy(y_pre, y_true)
        return lb + self.gama * loss1
