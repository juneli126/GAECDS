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
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import random
from model import *
from process import *

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from process import metric_scores




# load data
node_drug = pd.read_csv('./data/data_5693/drug_id_5693.csv')
features = pd.read_csv('./data/data_5693/feature197_300.csv')
cell_feature = pd.read_csv('./data/data_5693/cell_feature.csv')

drug1 = np.array(node_drug['g_id1'])
drug2 = np.array(node_drug['g_id2'])

cell_feature = cell_feature.drop(['cell'],axis=1)
cell_feature = np.array(cell_feature)
cell_feature = torch.tensor(cell_feature)
cell_feature = torch.reshape(cell_feature,(-1,954))
cell_feature = cell_feature.to(torch.float32)
print('cell_feature: ',cell_feature.shape)

node_feature = np.array(features)
node_feature = torch.tensor(node_feature)

node_feature = torch.reshape(node_feature, (-1, 300))
node_feature = node_feature.to(torch.float32)
print('node_feature: ',node_feature.shape)

drug_label = np.array(node_drug['label'])

adj = adj_create(drug1,drug2,drug_label,197)

adj = torch.tensor(adj)

# machine learning model
clf_svm = SVC(probability=True)
clf_rf = RandomForestClassifier()
clf_gbm = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
xgb.set_config(verbosity=0)


# mlp model
cell_mlp = MLP(in_feats=954,out_feats=128)

# GAE
# gae model
model = Model(300, 256, 128)
pos_weight = torch.tensor((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj = torch.tensor(adj)
labels = torch.reshape(adj, [-1])

opt_model = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.00001},
    {'params': cell_mlp.parameters(), 'lr': 0.01}
])

for epoch_gae in range(5):
    model.train()
    cell_mlp.train()
    pred_gcn,gcn_feature = model(adj, node_feature)
    pred_1 = torch.mul(labels,pred_gcn)

    print(pred_gcn)
    loss_model = norm * torch.mean(F.binary_cross_entropy_with_logits(input=pred_1, target=labels.float(), pos_weight=pos_weight))

    accuracy = acc(pred_1,labels)
    opt_model.zero_grad()
    loss_model.requires_grad_(True)
    loss_model.backward()
    opt_model.step()
    print('loss:',loss_model.item(),'accuracy:',accuracy)

    # gcn_cnn
    x_matrix = gcn_feature
    x_matrix = x_matrix.detach().numpy()

    cell_mlp.train()
    cell_out = cell_mlp(cell_feature,'sigmoid')
    cell_out = cell_out.detach().numpy()

    x_new_matrix = new_matrix_with_cell(drug1,drug2,cell_out,x_matrix)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

    y_values_svm = []
    predictions_svm = []
    probas_svm = []
    y_values_rf = []
    predictions_rf = []
    probas_rf = []
    y_values_gbm = []
    predictions_gbm = []
    probas_gbm = []
    y_values_xgb = []
    predictions_xgb = []
    probas_xgb = []
    X = np.array(x_new_matrix)
    y = drug_label
    for train, test in kf.split(X, y):
        # svm
        clf_svm.fit(X[train], y[train])
        predictions_svm.append(clf_svm.predict(X[test]))
        probas_svm.append(clf_svm.predict_proba(X[test]).T[1])  # Probabilities for class 1
        y_values_svm.append(y[test])

        # randomforest
        clf_rf.fit(X[train], y[train])
        predictions_rf.append(clf_rf.predict(X[test]))
        probas_rf.append(clf_rf.predict_proba(X[test]).T[1])  # Probabilities for class 1
        y_values_rf.append(y[test])

        # GBM
        clf_gbm.fit(X[train], y[train])
        predictions_gbm.append(clf_gbm.predict(X[test]))
        probas_gbm.append(clf_gbm.predict_proba(X[test]).T[1])  # Probabilities for class 1
        y_values_gbm.append(y[test])

        # xgboost
        clf_xgb.fit(X[train], y[train])
        predictions_xgb.append(clf_xgb.predict(X[test]))
        probas_xgb.append(clf_xgb.predict_proba(X[test]).T[1])  # Probabilities for class 1
        y_values_xgb.append(y[test])

    metric_scores(y_values_svm, probas_svm, predictions_svm)
    metric_scores(y_values_rf, probas_rf, predictions_rf)
    metric_scores(y_values_gbm, probas_gbm, predictions_gbm)
    metric_scores(y_values_xgb, probas_xgb, predictions_xgb)
