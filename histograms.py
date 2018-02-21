import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
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
def read_data():
    df = pd.read_csv('decor_extended.csv', encoding = "Latin-1")
    labels_country = df['country_label']
    labels_style = df['decor_label']
    labels_type = df['type_label']
    labels_style = labels_style.transpose()
    print(labels_style)
    return labels_country, labels_style, labels_type

"""read images from file"""
def read_images():
    data = np.zeros((3880, 768)) #number of files, number of clusters
    path = os.getcwd()
    path = path + "\\" + "images_extended"
    im_counter = 0;
    for root, dirs, files in os.walk(path):
        for file in files:
            impath = path + '\\' + file
            img = cv2.imread(impath)
            img = img.astype('float64')
            b,g,r = cv2.split(img)
            data[im_counter, 0:256] = make_histogram(r)
            data[im_counter, 256:512] = make_histogram(g)
            data[im_counter, 512:768] = make_histogram(b)                            
            im_counter += 1
    return data

"""make histograms"""
def make_histogram(img):
    #cut black edges from the image
    #apply mask so that only the patterns and interiors are taken into account
    hist, bins = np.histogram(img,bins=256)#
    hist = hist.reshape(-1,1)
    hist = hist.transpose()
    return hist

def split_data(data, labels):
    index = 0
    train = np.zeros((3000, 768))
    test = np.zeros((880, 768))
    labels_train = []
    labels_test = []
    #first split into product and pattern
    #then into different styles
    for i in range(max(labels)):
        #train = [data[i, :] for i == i]   #append collumns
        #remember to take part of products and part of patterns
        arr = len(np.extract(labels == 1, labels))
        #wrong, because the data needs to be appended to the bottom
        #make train and test as numpy arrays
        #probably good for labels
        train.append(data[index : index + arr - arr/5])
        test.append(data[index + arr - arr/5 : index + arr])
        index += arr    
    return train,test,labels_train,labels_test

def split_data_random(data, labels):
    
    return train,test,labels_train,labels_test

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
    labels_country, labels_style, labels_type = read_data()
    data = read_images()
    train,test,labels_train,labels_test = split_data(data, labels_style)
    train,test,labels_train,labels_test = split_data_random(data, labels_style)
    print(labels_train)
    model = train_svm(train, labels_train)
    predictions = predict(model, test)
    print(predictions)
    print(labels_test)
    #error = calculate_error(predictions, labels_test)
    
    #read_data_test()