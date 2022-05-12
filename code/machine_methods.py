import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,train_test_split,KFold,StratifiedKFold


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from process import metric_scores,multi_models_roc


# load data
data = pd.read_csv('./data/machine_methods/feature_1554.csv')

y = data['label']
X = data.drop(['label'],axis=1)
print(X.shape,y.shape)

X = np.array(X)
y = np.array(y)


clf_svm = SVC(probability=True)
clf_rf = RandomForestClassifier()
clf_gbm = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
xgb.set_config(verbosity=0)

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

tprs_svm = []
aucs_svm = []
mean_fpr = np.linspace(0, 1, 100)

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

print(probas_svm)
print(y_values_svm)

for i in range(5):
    for j in range(len(probas_svm[i])):
        with open('pred_svm%s.txt'%i,'a') as f_svm:
            f_svm.write(str(probas_svm[i][j]))
            f_svm.write('\t')
            f_svm.write(str(y_values_svm[i][j]))
            f_svm.write('\n')

        with open('pred_rf%s.txt'%i,'a') as f_rf:


            f_rf.write(str(probas_rf[i][j]))
            f_rf.write('\t')
            f_rf.write(str(y_values_rf[i][j]))
            f_rf.write('\n')

        with open('pred_gbm%s.txt'%i, 'a') as f_gbm:

            f_gbm.write(str(probas_gbm[i][j]))
            f_gbm.write('\t')
            f_gbm.write(str(y_values_gbm[i][j]))
            f_gbm.write('\n')

        with open('pred_xgb%s.txt'%i, 'a') as f_xgb:

            f_xgb.write(str(probas_xgb[i][j]))
            f_xgb.write('\t')
            f_xgb.write(str(y_values_xgb[i][j]))
            f_xgb.write('\n')



metric_scores(y_values_svm, probas_svm, predictions_svm)
metric_scores(y_values_rf, probas_rf, predictions_rf)
metric_scores(y_values_gbm, probas_gbm, predictions_gbm)
metric_scores(y_values_xgb, probas_xgb, predictions_xgb)
