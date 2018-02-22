import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd


"""
def read_data_test():
    with open('decor_extended.xlsx', encoding="Latin-1") as f:
        data_size = 3880
        data_matrix = np.zeros((data_size, 3))
        for i, line in enumerate(f):
            if i == 0:
                continue
            line_list = line.split(',')[:-1]
            data_matrix[i-1] = [int(line_list[j]) for j in [0, 2, 4]]
    #labels_country = data_matrix[:,0]
    print(data_matrix)
"""

    
"""read data from .csv file"""
def read_data(problem):
    path = os.getcwd()
    path_train = os.path.join(path, problem, 'training.csv')
    path_test = os.path.join(path, problem, 'testing.csv')
    train = pd.read_csv(path_train, encoding = "Latin-1")
    test = pd.read_csv(path_test, encoding = "Latin-1")
    labels_train = train.iloc[:,1]
    labels_test = test.iloc[:,1]
    #labels_train = labels.transpose()
    #labels_test = labels.transpose()
    print(labels_train) #3088
    print(labels_test) #792
    return labels_train, labels_test


"""read images from file"""
def read_images(problem):
    images_train = np.zeros((3088, 768)) #number of files, number of clusters
    images_test = np.zeros((792, 768))
    im_counter = 0
    #path = os.getcwd()
    for image_name in os.listdir(os.path.join(problem, 'training')):
        train = os.path.join(problem, 'training', image_name)
        print(train)
        row = each_image(train)
        images_train[im_counter, 0:768] = row                           
        im_counter += 1
    im_counter = 0
    for image_name in os.listdir(os.path.join(problem, 'testing')):
        test = os.path.join(problem, 'testing', image_name)
        print(test)
        row = each_image(test)
        images_test[im_counter, 0:768] = row                           
        im_counter += 1
    return images_train, images_test


def each_image(path):
    data = np.zeros((1, 768))
    img = cv2.imread(path)
    img = img.astype('uint8')
    b,g,r = cv2.split(img)
    data[0, 0:256] = make_histogram(r)
    data[0, 256:512] = make_histogram(g)
    data[0, 512:768] = make_histogram(b)
    return data


"""make histograms"""
def make_histogram(img):
    #apply mask so that only the patterns and interiors are taken into account
    hist, bins = np.histogram(img, bins = 256, range = (5,250))
    hist = hist.reshape(-1,1)
    hist = hist.transpose()
    return hist


def train_svm(training, labels):
    clf = SVC()
    clf.fit(training, labels) 
    return clf
    
def predict(model, test):
    predictions = model.predict(test)
    return predictions
    
def calculate_error(predicted, real):
    correct = 0
    for i in range(3880):
        if predicted[i] == real[i]:
            correct += 1
    accuracy = correct/3880
    print(accuracy)
    return accuracy

if __name__ == "__main__":  
    problem = "by_style"
    labels_train, labels_test = read_data(problem)
    images_train, images_test = read_images(problem)
    
    #train,test,labels_train,labels_test = split_data(data, labels_style)
    #train,test,labels_train,labels_test = split_data_random(data, labels_style)
    #print(labels_train)
    #model = train_svm(train, labels_train)
    #predictions = predict(model, test)
    #print(predictions)
    #print(labels_test)
    #error = calculate_error(predictions, labels_test)