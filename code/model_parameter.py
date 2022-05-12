import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
import itertools
import sklearn
from sklearn import svm
import torch.utils.data as Data
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import random
import matplotlib
matplotlib.use('Agg')


# Model
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        self.conv1 = dglnn.DenseGraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2 = dglnn.DenseGraphConv(
            in_feats=hid_feats, out_feats=out_feats)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, g_adj, feature,af1):

        h = self.conv1(g_adj, feature)
        if af1 == 'R':
            h = self.relu(h)
        elif af1 == 'S':
            h = self.sigmoid(h)
        else:
            h = self.tanh(h)
        h = self.conv2(g_adj, h)
        if af1 == 'R':
            h = self.relu(h)
        elif af1 == 'S':
            h = self.sigmoid(h)
        else:
            h = self.tanh(h)


        return h

class CNN(nn.Module):
    def __init__(self, in_feats, hid1_feats,hid2_feats, out_feats):
        super().__init__()
        self.hid2_feats = hid2_feats
        self.cnn1 = nn.Conv2d(in_channels=in_feats,out_channels=hid1_feats,kernel_size=(2,2),padding='same')
        self.bn1 = nn.BatchNorm2d(hid1_feats)
        self.cnn2 = nn.Conv2d(in_channels=hid1_feats, out_channels=hid2_feats,kernel_size=(2,2),padding='same')
        self.bn2 = nn.BatchNorm2d(hid2_feats)
        self.fc3 = nn.Linear(hid2_feats, out_feats)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x,af2):

        h = self.cnn1(x)
        h = self.bn1(h)
        if af2 == 'R':
            h = self.relu(h)
        elif af2 == 'S':
            h = self.sigmoid(h)
        else:
            h = self.tanh(h)

        #h = self.dropout(h)

        h = self.cnn2(h)
        h = self.bn2(h)
        if af2 == 'R':
            h = self.relu(h)
        elif af2 == 'S':
            h = self.sigmoid(h)
        else:
            h = self.tanh(h)

        #print(h.shape)
        #h = self.dropout(h)

        #h = self.fc3(h)
        #h = torch.sigmoid(h)
        h = h.view(-1,self.hid2_feats)
        h = self.fc3(h)
        #print(h.shape)
        h = self.sigmoid(h)
        #print(h.shape)

        return h

class InnerProductDecoder(nn.Module):
    def forward(self, inputs):

        x = inputs.T
        x = torch.mm(inputs, x)
        x = torch.reshape(x, [-1])
        outputs = torch.sigmoid(x)
        return outputs


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.encoder = GCN(in_features, hidden_features, out_features)
        self.decoder = InnerProductDecoder()
    def forward(self, adj, feature,af1):
        h = self.encoder(adj, feature,af1)
        #h_noise = h + (0.1**0.5)*torch.randn(187,128)
        h_noise = self.decoder(h)
        #print('h: ',h)
        return h_noise,h

class MLP(nn.Module):
    def __init__(self,in_feats,out_feats,hid1_feats=512,hid2_feats=256):
        super().__init__()

        self.mlp1 = nn.Linear(in_feats,hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats,hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats,out_feats)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.in_feats = in_feats

    def forward(self,x,method,af):

        x = x.view(-1,self.in_feats)
        out = self.mlp1(x)
        out = self.dropout(out)
        if af == 'R':
            out = self.relu(out)
        elif af == 'S':
            out = self.sigmoid(out)
        else:
            out = self.tanh(out)

        out = self.mlp2(out)
        out = self.dropout(out)
        if af == 'R':
            out = self.relu(out)
        elif af == 'S':
            out = self.sigmoid(out)
        else:
            out = self.tanh(out)
        out = self.mlp3(out)
        if method == 'relu':
            out = self.relu(out)
        elif method == 'sigmoid':
            out = self.sigmoid(out)

        return out