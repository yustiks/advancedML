import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
# Adaboost Accuracy to check product or pattern
#------------------------------------------------------------------------------
# Read Data from the folder
# Data Used:
    # 1.Images from decore_extended in HOG2.py. Output of HOG is 
    #   Stored in a csv file which contains the Hog values
    # 2.Used those as a data and then choose the labels from the decor_extended
    #   excel, extracted the labels to classify it as product or pattern
    #   similarlly can be done with other labels.        

data = pd.read_csv(r"decor_extended.csv")
HogValues = pd.read_csv(r"Hog_Values.csv")
#Ydata = data.iloc[:,0] #target values labels based on Country
#Ydata = data.iloc[:,2] #target values labels based on Style
Xdata = HogValues.iloc[:] #data-------Hog Values 
Ydata = data.iloc[:,4] #target values labels based on Style

#print((Xdata))
print((Ydata))
#------------------------------------------------------------------------------
#Random split of Dataset and feature
x_train, x_test, y_train, y_test = train_test_split(Xdata,Ydata, test_size=0.2, random_state=0)
#------------------------------------------------------------------------------
#==============================================================================
# Training Data by Country/Product/Style from folder
#   Below section specific hardcode split of train and test for the data by 
#   country. Kindly change it according to the targets of the specific class.
#   Data used is HOG features.
#==============================================================================
#x_train = Xdata[:3095]
#x_test = Xdata[3096:3879]
#print(x_test)
#y_csvtrain = pd.read_csv(r"C:\Users\aswin\Desktop\advancedML\by_country\training.csv")
#y_csvtest = pd.read_csv(r"C:\Users\aswin\Desktop\advancedML\by_country\testing.csv")
#y_train =y_csvtrain.iloc[:,1] 
#y_test = y_csvtest.iloc[:,1]
#print(y_train)
#print(x_train)

#------------------------------------------------------------------------------
#print(x_train)
#print(y_train)
#------------------------------------------------------------------------------
# DecisionTree
features_in_label = 2; #according to the label
 
Dt = DecisionTreeClassifier()
Dt.fit(x_train,y_train)
Accuracy_decisionTree = Dt.score(x_test,y_test)
Dt_err = 1.0 - Dt.score(x_test, y_test)

print('Decision Tree')
print(Accuracy_decisionTree)
#print('Error Decision Tree')
#print(Dt_err)
# Ramdom Forest
RForest = RandomForestClassifier(n_estimators=50,max_features = features_in_label)
RForest.fit(x_train,y_train)
Accuracy_RandomForest = RForest.score(x_test,y_test)
print('Random Forest ')
print(Accuracy_RandomForest)
#------------------------------------------------------------------------------
# AdaBoost
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100,
                              learning_rate=0.1,algorithm='SAMME.R', 
                              random_state=None)
AdaBoost.fit(x_train,y_train)
Accuracy_AdaBoost = AdaBoost.score(x_test,y_test)
Accuracy_AdaBoostTrain = AdaBoost.score(x_train,y_train)
print('AdaBoost ')
print(Accuracy_AdaBoost)
#------------------------------------------------------------------------------
# GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators = 100,max_features = features_in_label)
GBC.fit(x_train,y_train)
Accuracy_GradBoost = GBC.score(x_test,y_test)
print('GradientBoosting ')
print(Accuracy_GradBoost)