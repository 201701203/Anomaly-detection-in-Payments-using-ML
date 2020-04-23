# import necessary libraries
import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import train_test_split

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

# split the data for training 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#--------Plot the data-----------------


#--------------------------------1---Isolation Forest----------------------------------#
F = iso.iForest(X=x,ntrees=500, sample_size=256)    #Create Forest
S=F.compute_paths(X_in=x)


#--------------------------------2---Local outlier Factor------------------------------#
y_lof = lof.LocalOutlierFactor(X=x, k=10)
lof_accuracy = (y_lof == y_test).mean()

#--------------------------------3---Logistic Regression-------------------------------#
model = log_reg.LogisticRegression(lr=0.1, num_iter=120000) # Learning rate = 0.1 and Number of Iteration 120000
model.fit(x_train, y_train)
preds = model.predict(x_test)
log_reg_accuracy = (preds == y_test).mean()

#--------------------------------4---Support vector machine----------------------------#
