# import necessary libraries
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# import model
import Isolation_Forest as iso
import Local_Outlier_Factor
import Logistic_Regression
import Support_Vector_Machine

#read data from excel (Data-set)
data=pd.read_excel('creditcard.xlsx')   # https://www.kaggle.com/mlg-ulb/creditcardfraud
x=np.array([data.iloc[1:, 1:29]]).T     
y=np.array([data.iloc[1:,30]]).T        # target class

#for making it normal
x = x-np.mean(x)
x = x/np.std(x)


#--------------------------------1---Isolation Forest----------------------------------#
F = iso.iForest(X=x,ntrees=500, sample_size=256)    #Create Forest
S=F.compute_paths(X_in=X)


#--------------------------------2---Local outlier Factor------------------------------#



#--------------------------------3---Logistic Regression-------------------------------#



#--------------------------------4---Support vector machine----------------------------#

