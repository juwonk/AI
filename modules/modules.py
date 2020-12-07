import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

def LoadINPUT(file):
    D1 = pd.read_csv(file, index_col=0)
    return(D1.iloc[:-1,:], D1.loc['Group',:]) #return : (x, y) y~last column named 'Group' & [0.1/0.9]

def FeatureStats(data, label, pval = 0.05):    
    TaxaIDs = []
    Levens = []
    Equality = []
    TPs = []
    Avg = []

    for i in range(data.shape[0]):
        D_i = data.iloc[i,:]

        #shaP = stats.shapiro(D_i)[1]
        levP = stats.levene(D_i[label == 0.1], D_i[label == 0.9])[1]
        #print(shaP, levP)
        equl = (levP > pval)
        if(equl): #두 그룹의 분산이 같지 않음 -> equal_var=False
            Pval = stats.ttest_ind(D_i[label == 0.1], D_i[label == 0.9], equal_var=True)[1]
        else:
            Pval = stats.ttest_ind(D_i[label == 0.1], D_i[label == 0.9], equal_var=False)[1]

        TaxaIDs.append(D_i.name)
        Levens.append(levP)
        Equality.append(equl)
        TPs.append(Pval)
        Avg.append(D_i.mean())

    TestResDic = {'Taxa_ID':TaxaIDs, 'Leven_Pval':Levens, 'Equal':Equality, 'Ttest_Pval':TPs, 'Average':Avg}
    TestResDF = pd.DataFrame(TestResDic).sort_values('Average', ascending=False)
    
    return(TestResDF)