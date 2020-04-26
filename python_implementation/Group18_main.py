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
F = iso.iForest(X=x,ntrees=50, sample_size=25)
y_iso=F.predict(X_in=x)
iso_accuracy = (y_svm == y).mean()
iso_accuracy

#--------------------------------2---Local outlier Factor------------------------------#
y_lof = lof.LocalOutlierFactor(X=x, k=10, outlier_threshold = 4)
lof_accuracy = (y_lof == y).mean()
lof_accuracy

#--------------------------------3---Logistic Regression-------------------------------#
model = log_reg.LogisticRegression(lr=0.01, num_iter=1000)
model.fit(x_train, y_train)
preds = model.predict(x_test)
log_reg_accuracy = (preds == y_test).mean()
log_reg_accuracy

#--------------------------------4---Support vector machine----------------------------#
model_svm = svm.SVM()
model_svm.fit(X = x_train, y=y_train)
y_svm = model_svm.predict(x_test)
svm_accuracy = (y_svm == y_test).mean()
svm_accuracy
