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
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,f1_score,recall_score,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import random
import matplotlib
matplotlib.use('Agg')


def adj_create(src, dst,label,num):
    adj_x = np.zeros(shape=(num,num))
    for i in range(0,num):
        adj_x[i][i] = 1
    for i in range(0, len(src)):
        if label[i]:
            src_i1 = int(src[i])
            dst_j1 = int(dst[i])

            adj_x[src_i1][dst_j1] = 1
            adj_x[dst_j1][src_i1] = 1

    return adj_x

def acc(pred, labels):
    count = 0
    a = torch.where(pred >0.99,1,0).type(torch.int32)
    b = labels.type(torch.int32)
    for i in torch.eq(a,b):
        if i:
            count += 1

    acc_all = count / len(pred)
    return acc_all

def print_pred(pred):
    pred_np1 = pred.detach().numpy()
    for i in range(len(pred_np1)):
        if pred_np1[i] >= 0.99:
            pred_np1[i] = 1
        else:
            pred_np1[i] = 0
    pred_tensor = torch.tensor(pred_np1)
    pred_new1 = torch.reshape(pred_tensor,[679,679])

    return pred_new1

def new_matrix(src,drt,matrix):
    new_matrix_x = []
    for i in range(len(src)):
        x_i = src[i]
        x_j = drt[i]
        x_new = matrix[x_i] + matrix[x_j]

        new_matrix_x.append(x_new)
    return new_matrix_x

def new_matrix_val(src,drt,val,label,matrix):
    new_matrix_x = []
    new_label = []
    val_matrix = []
    val_id_i = []
    val_id_j = []
    for i in range(len(src)):
        if val[i] == 0:
            new_label.append(label[i])

            x_i = src[i]
            x_j = drt[i]
            x_new = matrix[x_i] + matrix[x_j]
            new_matrix_x.append(x_new)
        elif val[i] == 1:
            x_i = src[i]
            x_j = drt[i]
            x_new = matrix[x_i] + matrix[x_j]
            val_id_i.append(x_i)
            val_id_j.append(x_j)
            val_matrix.append(x_new)

    return new_matrix_x,new_label,val_matrix,val_id_i,val_id_j


def new_matrix_with_cell(src,drt,cell,matrix):
    new_matrix_x = []
    for i in range(len(src)):
        x_i = src[i]
        x_j = drt[i]
        x_cell = cell[i]
        x_new = matrix[x_i] + matrix[x_j] + x_cell

        new_matrix_x.append(x_new)
    return new_matrix_x

def negative_analysic(matrix,label,length):

    matrix_1_new = []
    label_1_new= []
    matrix_0_new = []
    label_0_new = []

    for i in range(len(label)):
        if label[i] == 1:
            matrix_1_new.append(matrix[i])
            label_1_new.append(label[i])
        elif label[i] == 0:
            matrix_0_new.append(matrix[i])
            label_0_new.append(label[i])
    #print(matrix_1_new)
    #print(matrix_0_new)

    matrix_0_all = matrix_0_new[0:length]
    #print(matrix_0_all)
    label_0 = label_0_new[0:length]
    matrix_all = matrix_0_all + matrix_1_new
    #print(matrix_all)
    label_all = label_0 + label_1_new

    new_matrix_all = []
    new_label_all = []
    lenth = len(matrix_all)
    random_num = random.sample(range(0, lenth), lenth)
    for i in range(len(random_num)):
        new_matrix_all.append(matrix_all[random_num[i]])
        new_label_all.append(label_all[random_num[i]])


    return new_matrix_all,new_label_all




def metrics_draw(true,pred,name):

    true = true.detach().numpy()
    pred = pred.detach().numpy()

    Truelist = []
    Problist = []
    for i in range(len(true)):
        Truelist.append(true[i][0])
        Problist.append(pred[i][0])

    Problist_int = []
    for i in range(len(Problist)):
        if Problist[i] >= 0.5:
            Problist_int.append(1)
        else:
            Problist_int.append(0)
    #print(Problist_int)


    precision_scores = metrics.precision_score(Truelist,Problist_int)
    recall_scores = metrics.recall_score(Truelist,Problist_int)
    f1_scores = metrics.f1_score(Truelist,Problist_int)
    print('f1_scores:', f1_scores)
    print('precision_scores:', precision_scores)
    print('recall_scores:', recall_scores)

    with open('metrics.txt','a') as f:
        f.write('f1_scores:')
        f.write(str(f1_scores))
        f.write('\r\n')
        f.write('precision_scores:')
        f.write(str(precision_scores))
        f.write('\r\n')
        f.write('recall_scores:')
        f.write(str(recall_scores))
        f.write('\r\n')


    precision,recall,_ = metrics.precision_recall_curve(Truelist,Problist)
    pr_auc = metrics.auc(recall,precision)
    print('pr_auc:', pr_auc)

    plt.figure(1)
    plt.plot(recall, precision, 'g', label='AUPR = %0.4f' % pr_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.savefig('./%sAUPR'%name+'.jpg')

    fpr, tpr, thresholds = metrics.roc_curve(Truelist, Problist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('roc_auc:',roc_auc)

    plt.figure(2)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig('./%sAUC'%name+'.jpg')
    #plt.show()


def metric_scores(y_values_all,probas_all,predictions_all):

    #print(y_values_all)
    #print(predictions_all)

    aucs = [roc_auc_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    accs = [accuracy_score(y, pred) for y, pred in zip(y_values_all, predictions_all)]

    for prob_i in probas_all:
        # print(prob_i)
        for i in range(len(prob_i)):
            # print(prob_i[i])
            if prob_i[i] > 0.5:
                prob_i[i] = 1
            else:
                prob_i[i] = 0


    precision_scores = [precision_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    recall_scores = [recall_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    f1_scores = [f1_score(y, proba) for y, proba in zip(y_values_all, probas_all)]

    pr_all = []
    for i in range(len(predictions_all)):

        precision, recall, _ = metrics.precision_recall_curve(y_values_all[i],predictions_all[i])
        pr_auc = metrics.auc(recall, precision)
        pr_all.append(pr_auc)

    #pr_auc = metrics.auc(np.mean(recall_scores), np.mean(precision_scores))

    print('accuracy: ',np.mean(accs),accs)
    print('roc_auc :',np.mean(aucs),aucs)
    print('pr_auc: ',np.mean(pr_auc),pr_all)
    print('precision_scores: ',np.mean(precision_scores),precision_scores)
    print('recall scores: ',np.mean(recall_scores),recall_scores)
    print('f1_scores: ',np.mean(f1_scores),f1_scores)

def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
    
    plt.figure(figsize=(20, 20), dpi=dpin)

    for (name, method, colorname) in zip(names, sampling_methods, colors):

        y_test_preds = method.predict(X_test)
        y_test_predprob = method.predict_proba(X_test)[: ,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)) ,color = colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.axis('square')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate' ,fontsize=20)
        plt.ylabel('True Positive Rate' ,fontsize=20)
        plt.title('ROC Curve' ,fontsize=25)
        plt.legend(loc='lower right' ,fontsize=20)

    if save:
        plt.savefig('multi_models_roc.png')

    return plt