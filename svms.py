import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import rbf_kernel
import cv2
import os
import pandas as pd
   
"""read histograms"""
def read_hist(problem, bins):
    train_path = os.path.join(problem,'training_hsv_histograms_{}.dat'.format(bins))
    hist_train = np.load(train_path)
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


def svm(training, labels, test, real):
    #make lists of different parameters for SVC, then iterate through some of them
    c = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 200, 300]
    kern = ['rbf','poly','linear']
    for i in range(len(kern)):
        for j in c:             
            model = SVC(C = j, kernel = kern[i], max_iter = -1)
            model.fit(training, labels) 
            accuracy = model.score(test, real)
            print("kernel: ", kern[i], ", C: ", j, ", accuracy: ", accuracy)


def svm_nu(training, labels, test, real):
    #make lists of different parameters for SVC, then iterate through some of them
    kern = ['rbf','poly','linear']
    for i in range(len(kern)):
        model = NuSVC(kernel = kern[i], nu = 0.38, degree = 3, gamma = 0.00005, coef0 = 1)
        model.fit(training, labels) 
        accuracy = model.score(test, real)
        print("kernel: ", kern[i], ", accuracy: ", accuracy)


def svm_chi2(training, labels, test, real):
    chi2 = chi2_kernel(training, gamma = 0.02)
    chi2_test = chi2_kernel(test, training, gamma = 0.02)
    model = SVC(C = 1, kernel = 'precomputed',max_iter = -1)
    model.fit(chi2, labels) 
    accuracy = model.score(chi2_test, real)
    print(accuracy)


def svm_laplacian(training, labels, test, real):
    laplacian = laplacian_kernel(training)
    laplacian_test = laplacian_kernel(test, training)
    model = SVC(C = 4, kernel = 'precomputed', max_iter = -1)
    model.fit(laplacian, labels) 
    accuracy = model.score(laplacian_test, real)
    print(accuracy)


def svm_rbf(training, labels, test, real):
    rbf = rbf_kernel(training, gamma = 0.0059)
    rbf_test = rbf_kernel(test, training, gamma = 0.0059)
    model = SVC(C = 4, kernel = 'precomputed', max_iter = -1)
    model.fit(rbf, labels) 
    accuracy = model.score(rbf_test, real)
    print(accuracy)


if __name__ == "__main__":  
    
    problems = ['by_country', 'by_style', 'by_product']
    bins = ['256', '128', '064', '032', '016', '010']    
    hist_train, hist_test = read_hist(problems[1], bins[0])
    labels_train, labels_test = read_labels(problems[1], bins[0])
    svm_chi2(hist_train, labels_train, hist_test, labels_test)
    svm_laplacian(hist_train, labels_train, hist_test, labels_test)
    svm_rbf(hist_train, labels_train, hist_test, labels_test)
    #svm(hist_train, labels_train, hist_test, labels_test)
    #svm_nu(hist_train, labels_train, hist_test, labels_test)

#chi2 with 10 bins, gamma 0.7
#laplacian hsv with 32 bins C = 1.1