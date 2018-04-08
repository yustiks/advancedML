#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# knn on hog vallues

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv(r"decor_extended.csv")
HogValues = pd.read_csv(r"Hog_Values.csv")
Ydata_country = data.iloc[:,0] #target values labels based on Country
Ydata_style = data.iloc[:,2] #target values labels based on Style
Xdata = HogValues.iloc[:] #data-------Hog Values 
Ydata_product = data.iloc[:,4] #target values labels based on Product

x_train, x_test, y_train_product, y_test_product = train_test_split(Xdata,Ydata_product, test_size=0.2, random_state=0)
x_train, x_test, y_train_country, y_test_country = train_test_split(Xdata,Ydata_country, test_size=0.2, random_state=0)
x_train, x_test, y_train_style, y_test_style = train_test_split(Xdata,Ydata_style, test_size=0.2, random_state=0)


def knn1(training, labels, test, real, name, k):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #model = KNeighborsClassifier(n_neighbors=1, weights='uniform')
    #model = DecisionTreeClassifier()
    model.fit(training, labels) 
    accuracy = model.score(test, real)
    print(name+': '+str(accuracy))
    
    #predicted = model.predict(test)
    #print(metrics.classification_report(real, predicted))
    #print(metrics.confusion_matrix(real, predicted))
        

if __name__ == "__main__": 
    for i in range(1, 11):
        print('k=',str(i))
        knn1(x_train, y_train_product, x_test, y_test_product, 'by_product',i)
        knn1(x_train, y_train_country, x_test, y_test_country, 'by_country',i)
        knn1(x_train, y_train_style, x_test, y_test_style, 'by_style',i)
            