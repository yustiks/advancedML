#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv(r"decor_extended.csv")
HogValues = pd.read_csv(r"Hog_Values.csv")
#Ydata = data.iloc[:,0] #target values labels based on Country
#Ydata = data.iloc[:,2] #target values labels based on Style
Xdata = HogValues.iloc[:] #data-------Hog Values 
Ydata = data.iloc[:,4] #target values labels based on Product

x_train, x_test, y_train, y_test = train_test_split(Xdata,Ydata, test_size=0.2, random_state=0)

def knn1(training, labels, test, real):
    model = KNeighborsClassifier(n_neighbors=2, weights='distance')
    #model = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    #model = DecisionTreeClassifier()
    model.fit(training, labels) 
    accuracy = model.score(test, real)
    print(accuracy)
    
    #predicted = model.predict(test)
    #print(metrics.classification_report(real, predicted))
    #print(metrics.confusion_matrix(real, predicted))
        

if __name__ == "__main__": 

    knn1(x_train, y_train, x_test, y_test)
            