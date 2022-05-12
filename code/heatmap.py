import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable
from sklearn import datasets
import pylab
import io
from PIL import Image


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def heatmap_draw(x_tick, y_tick,labels, X1, name1,name2,cell,name,label=True):

    data1 = {}
    for i in range(3):
        data1[x_tick[i]] = X1[i]
    pd_data1 = pd.DataFrame(data1, index=y_tick, columns=x_tick)



    plt.figure(dpi=1200)

    if label:
        sns.heatmap(data=pd_data1, cmap='YlGnBu',annot=labels, fmt='')
    else:
        sns.heatmap(data=pd_data1, cmap='YlGnBu', annot=True, fmt='')

    plt.xlabel(name1, fontsize=20, color='k')
    plt.ylabel(name2, fontsize=20, color='k')
    plt.title(cell)

    png1 = io.BytesIO()
    plt.savefig(png1, format="tiff", dpi=1200, pad_inches=.1, bbox_inches='tight')
    # png2 = Image.open(png1)
    plt.savefig('./' + name + '.tiff')



x1_tick = ['0','0.5','1']
x2_tick = ['0','0.25','0.5']
y1_tick = ['10','5','0']
X1 = [[0.114,0.085,0],[0.68,0.581,0.201],[0.779,0.764,0.325]]
X2 = [[0.477,0.382,0],[0.756,0.713,0.516],[0.777,0.720,0.607]]

heatmap_draw(x2_tick,y1_tick,labels1,X1,'Mitoxantrone','Trametinib','Hela','M+T_hela',label=False)
heatmap_draw(x1_tick,y1_tick,labels1,X2,'Mitoxantrone','Trametinib','HepG2','M+T_hepg2',label=False)

y3_tick = ['4','2','0']
X3 = [[0.364,0.27,0],[0.575,0.58,0.201],[0.62,0.6,0.325]]
X4 = [[0.308,0.183,0],[0.45,0.481,0.379],[0.495,0.487,0.447]]

heatmap_draw(x2_tick,y3_tick,labels2,X3,'Mitoxantrone','Capivasertib','Hela','M+C_hela',label=False)
heatmap_draw(x1_tick,y3_tick,labels3,X4,'Mitoxantrone','Capivasertib','HepG2','M+C_hepg2',label=False)

y4_tick = ['0.5','0.25','0']
X5 = [[0.10,0.04,0],[0.45,0.56,0.317],[0.525,0.48,0.428]]
labels4 = np.array([[0.10,0.45,0.525],[0.04,0.56,'NaN'],[0,0.317,0.428]])

heatmap_draw(x1_tick,y4_tick,labels2,X5,'Mitoxantrone','Flumethasone','Hela','M+F_hela',label=False)

y5_tick = ['80','40','20','0']
y6_tick = ['80','60','40','0']
x3_tick = ['0','0.25','0.5','1']
X6 = [[0.393,0.294,0.213,0],[0.5,0.468,0.408,0.273],[0.517,0.571,0.45,0.307]]
X7 = [[0.178,0.112,0.082,0],[0.51,0.475,0.471,0.3],[0.55,0.478,0.494,0.357],[0.61,0.612,0.57,0.453]]

heatmap_draw(x2_tick,y5_tick,labels6,X6,'Mitoxantrone','Methylprednisolone','Hela','M+M_hela',label=False)
heatmap_draw(x3_tick,y6_tick,labels7,X7,'Mitoxantrone','Methylprednisolone','HepG2','M+M_hepg2',label=False)







