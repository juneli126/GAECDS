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
from model_parameter import *
from process import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
#drug_label = torch.tensor(drug_label)
#labels = torch.reshape(drug_label, [-1])
#drug_label.to(torch.int64)

adj = adj_create(drug1,drug2,drug_label,197)

adj = torch.tensor(adj)

# parameter

parameter_list = [[0.01, 0.0001, 0.001, 'R', 'R', 'R', 128],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'R', 128],
                  [0.0001, 0.0001, 0.001, 'R', 'R', 'R', 128],
                  [0.00001, 0.0001, 0.001, 'R', 'R', 'R', 128],
                  [0.001, 0.01, 0.001, 'R', 'R', 'R', 128],
                  [0.001, 0.001, 0.001, 'R', 'R', 'R', 128],
                  [0.001, 0.00001, 0.001, 'R', 'R', 'R', 128],
                  [0.001, 0.0001, 0.01, 'R', 'R', 'R', 128],
                  [0.001, 0.0001, 0.0001, 'R', 'R', 'R', 128],
                  [0.001, 0.0001, 0.00001, 'R', 'R', 'R', 128],
                  [0.001, 0.0001, 0.001, 'S', 'R', 'R', 128],
                  [0.001, 0.0001, 0.001, 'T', 'R', 'R', 128],
                  [0.001, 0.0001, 0.001, 'R', 'S', 'R', 128],
                  [0.001, 0.0001, 0.001, 'R', 'T', 'R', 128],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'S', 128],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'T', 128],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'R', 32],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'R', 64],
                  [0.001, 0.0001, 0.001, 'R', 'R', 'R', 256],]


for i_param in range(len(parameter_list)):
    lr_gcn = parameter_list[i_param][0]
    lr_cnn = parameter_list[i_param][1]
    lr_mlp = parameter_list[i_param][2]
    af_gcn = parameter_list[i_param][3]
    af_cnn = parameter_list[i_param][4]
    af_mlp = parameter_list[i_param][5]
    vector_d = parameter_list[i_param][6]
    print('parameters:%s_%s_%s_%s_%s_%s_%s ' % (lr_gcn, lr_cnn, lr_mlp, af_gcn, af_cnn, af_mlp, vector_d))


    # cnn and  mlp model
    net = CNN(vector_d,64,32,1)
    cell_mlp = MLP(in_feats=954,out_feats=vector_d)

    opt_cnn = torch.optim.Adam([
        {'params': net.parameters(), 'lr': lr_cnn},
        {'params': cell_mlp.parameters(), 'lr': lr_mlp}
    ])

    # GAE
    # gae model
    model = Model(300, 256, vector_d)
    pos_weight = torch.tensor((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj = torch.tensor(adj)
    labels = torch.reshape(adj, [-1])

    opt_model = torch.optim.Adam(model.parameters(),lr=lr_gcn)
    for epoch_gae in range(5):
        model.train()
        pred_gcn,gcn_feature = model(adj, node_feature,af_gcn)
        pred_1 = torch.mul(labels,pred_gcn)

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


        cell_out = cell_mlp(cell_feature,'sigmoid',af_mlp)
        cell_out = cell_out.detach().numpy()

        x_new_matrix = new_matrix_with_cell(drug1, drug2, cell_out, x_matrix)

        x_new_matrix = torch.tensor(x_new_matrix)
        x_new_matrix = torch.reshape(x_new_matrix, (-1, vector_d, 1, 1))

        # five fold cross validation
        lenth = len(x_new_matrix)
        pot = int(lenth / 5)
        print('lenth', lenth)
        print('pot', pot)

        random_num = random.sample(range(0, lenth), lenth)
        #print(random_num)
        for i_time in range(5):

            test_num = random_num[pot * i_time:pot * (i_time + 1)]
            train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

            x_train = x_new_matrix[train_num]
            x_test = x_new_matrix[test_num]

            y_train = drug_label[train_num]
            y_test = drug_label[test_num]


            y_train = torch.tensor(y_train)
            y_test = torch.tensor(y_test)
            y_train = torch.reshape(y_train, (-1, 1))
            y_test = torch.reshape(y_test, (-1, 1))

            batch_size = 128
            torch_dataset = Data.TensorDataset(x_train, y_train)
            loader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            for epoch_cnn in range(200):
                # train
                net.train()
                cell_mlp.train()
                for step, (batch_x, batch_y) in enumerate(loader):
                    pred_cnn = net(batch_x,af_cnn)

                    pred_new = pred_cnn
                    batch_y_new = batch_y
                    pred_cnn = pred_cnn.to(torch.float32)
                    batch_y = batch_y.to(torch.float32)
                    loss_net = torch.mean(F.binary_cross_entropy(pred_cnn, batch_y))

                    acc_train = acc(pred_new, batch_y_new)

                    loss_net.requires_grad_(True)
                    opt_cnn.zero_grad()
                    loss_net.backward()
                    opt_cnn.step()
                   

                # test
                net.eval()
                cell_mlp.eval()
                pred_test = net(x_test,af_cnn)
                acc_test = acc(pred_test, y_test)
                #print('y_test: ',y_test)
                #print('pred_test: ',pred_test)

                print('parameters:%s_%s_%s_%s_%s_%s_%s '%(lr_gcn,lr_cnn,lr_mlp,af_gcn,af_cnn,af_mlp,vector_d),'epoch_GAE: ',epoch_gae, 'cross validation: ', i_time, 'Epoch: ', epoch_cnn, '|loss: ', loss_net.item(), '| accuracy_train: ',
                      acc_train, '| accuracy_test: ', acc_test)


            metrics_scores = metrics_draw(y_test, pred_test, 'cell_epoch%s_%s' % (epoch_gae,i_time))



# save model

torch.save(net,'net_cnn_param.pt')
print('Save model!')








