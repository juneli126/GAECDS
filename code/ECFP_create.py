from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import DataStructs

import pandas as pd

data = pd.read_csv('./data/nagetive_analysic/drug_id_5376.csv')

X_1 = data['smiles']
print(X_1.shape)

for i in range(0,int(X_1.shape[0])):
    m1 = X_1[i]
    #print(m1)
    #name1 = name[i]
    m1 = Chem.MolFromSmiles(m1)
    print(i)
    #print(name1)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 6, nBits=300)
    f = open('drug_5376.txt', 'a')
    f.write(fp1.ToBitString())
    f.write('\r\n')
    f.close()
    print(fp1.ToBitString())
