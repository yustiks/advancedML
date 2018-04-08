#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# knn on histograms_rgb and histograms_hsv

import numpy as np
import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

"""read histograms"""
def read_hist(problem, bins):
#    train_path = os.path.join(problem,'training_rgb_histograms_{}.dat'.format(bins))
    train_path = os.path.join(problem,'training_hsv_histograms_{}.dat'.format(bins))
    hist_train = np.load(train_path)
#    test_path = os.path.join(problem,'testing_rgb_histograms_{}.dat'.format(bins))
    test_path = os.path.join(problem,'testing_hsv_histograms_{}.dat'.format(bins))
    hist_test = np.load(test_path)
    return hist_train, hist_test

"""read labels"""
def read_labels(problem, bins):
    train_path = os.path.join(problem,'training_labels_{}.dat'.format(bins))
    labels_train = np.load(train_path)
    test_path = os.path.join(problem,'testing_labels_{}.dat'.format(bins))
    labels_test = np.load(test_path)
    return labels_train, labels_test

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

    problems = ['by_country', 'by_style', 'by_product']
    bins = ['256', '128', '064', '032', '016', '010']
    
    for i in range(len(problems)):    
        problem = problems[i]
        print (problem)
        for j in range(len(bins)):    
            colors_in_histogram = bins[j]
            print(colors_in_histogram)
            hist_train, hist_test = read_hist(problem, colors_in_histogram)
            labels_train, labels_test = read_labels(problem, colors_in_histogram)
            knn1(hist_train, labels_train, hist_test, labels_test)