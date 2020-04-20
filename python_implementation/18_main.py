# import necessary libraries
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# import model
import Isolation_Forest as iso
import Local_Outlier_Factor as lof
import Logistic_Regression as log_reg
import Support_Vector_Machine as svm

#read data from excel (Data-set)
data=pd.read_csv(r"C:\\Users\\hp\\Desktop\\ML\\creditcard.csv") 
x=data.iloc[1:, 1:29]     
y=data.iloc[1:,30]       # target class

#for making it normal
x = x-np.mean(x)
x = x/np.std(x)

#--------Plot the data-----------------


#--------------------------------1---Isolation Forest----------------------------------#
F = iso.iForest(X=x,ntrees=500, sample_size=256)    #Create Forest
S=F.compute_paths(X_in=x)


#--------------------------------2---Local outlier Factor------------------------------#



#--------------------------------3---Logistic Regression-------------------------------#
model = log_reg.LogisticRegression(lr=0.1, num_iter=120000) # Learning rate = 0.1 and Number of Iteration 120000
model.fit(x, y)
preds = model.predict(X)
model.theta


#--------------------------------4---Support vector machine----------------------------#
