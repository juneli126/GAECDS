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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import random
from matplotlib.lines import Line2D
import io
from PIL import Image

def AUC_draw(data,label,color,save=False):

    data = np.array(data)
    model_all = []
    auc_all = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(5):
        idx_p = 2 * i
        idx_t = idx_p + 1
        data_p = data[idx_p]
        data_t = data[idx_t]

        fpr, tpr, thresholds = metrics.roc_curve(data_t, data_p, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        auc_all.append(roc_auc)
        model_all.append(np.interp(mean_fpr, fpr, tpr))
        model_all[-1][0] = 0.0

    plt.figure(1)

    mean_tpr = np.mean(model_all, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc_all)
    
    plt.plot(mean_fpr, mean_tpr, color=color, label= label+' (AUC = %0.4f)' % mean_auc, lw=2, alpha=.8)
    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="tiff", dpi=1200, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    if save:
        png2.save("./AUC_all.tiff")
    png1.close()


data_gbm = pd.read_csv('./result/machine/gbm_all.csv')
data_rf = pd.read_csv('./result/machine/rf_all.csv')
data_svm = pd.read_csv('./result/machine/svm_all.csv')
data_xgb = pd.read_csv('./result/machine/xgb_all.csv')
data_GAECDS = pd.read_csv('./result/deep_methods/GAECDS/cell/GAECDS_all.csv')
data_DeepDDS = pd.read_csv('./result/deep_methods/DeepDDS/deepdds_all.csv')
data_DeepSynergy = pd.read_csv('./result/deep_methods/DeepSynergy/DeepSynergy_all.csv')


AUC_draw(data_svm,'SVM','steelblue',save=False)
AUC_draw(data_rf,'RF','orange',save=False)
AUC_draw(data_gbm,'GBM','mediumpurple',save=False)
AUC_draw(data_xgb,'XGB','mediumseagreen',save=False)

AUC_draw(data_DeepDDS,'DeepDDS','lime',save=False)
AUC_draw(data_DeepSynergy,'DeepSynergy','purple',save=False)
AUC_draw(data_GAECDS,'GAECDS','crimson',save=True)

