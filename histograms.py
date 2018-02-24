import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import cv2
import os
import sys
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
    # print(labels_train) #3088
    # print(labels_test) #792
    return labels_train, labels_test


"""read images from file"""
def read_images(problem, training_size, testing_size, clusters_size):

    images_train = np.zeros((training_size, 3 * clusters_size))
    images_test = np.zeros((testing_size, 3 * clusters_size))
    im_counter = 0

    for image_name in os.listdir(os.path.join(problem, 'training')):
        print('{0} - training - {1}'.format(clusters_size, image_name))
        train = os.path.join(problem, 'training', image_name)
        # print(train)
        row = each_image(train, clusters_size)
        images_train[im_counter, :] = row                           
        im_counter += 1
    im_counter = 0

    for image_name in os.listdir(os.path.join(problem, 'testing')):
        print('{0} - testing - {1}'.format(clusters_size, image_name))
        test = os.path.join(problem, 'testing', image_name)
        # print(test)
        row = each_image(test, clusters_size)
        images_test[im_counter, :] = row                           
        im_counter += 1

    return images_train, images_test


def each_image(path, clusters_size):
    data = np.zeros((1, 3 * clusters_size))
    img = cv2.imread(path)
    img = img.astype('uint8')

	# get rid of background whites and blacks 
    black_threshold = 4
    white_threshold = 251

    filtered_img = [e for e in img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
         if (np.all(e > black_threshold) and np.all(e < white_threshold))]
    recomposed_img = np.expand_dims(np.asarray(filtered_img), axis=0)

    b,g,r = cv2.split(recomposed_img)
    data[0, 0:clusters_size] = make_histogram(r, clusters_size)
    data[0, clusters_size:2*clusters_size] = make_histogram(g, clusters_size)
    data[0, 2*clusters_size:] = make_histogram(b, clusters_size)

    return data


"""make histograms"""
def make_histogram(img, clusters_size):
    # apply mask so that only the patterns and interiors are taken into account
    hist, bins = np.histogram(img, bins=clusters_size)
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

def serialize_histograms(problem, dataset_type, dataset, clusters_size):
	path = os.path.join(problem, dataset_type + \
	 ('_histograms_%.3d.dat' % clusters_size))
	dataset.dump(path)
	

if __name__ == "__main__":  
    
    dataset_sizes = {
        'by_country' : (3096, 784),
        'by_product' : (3096, 784),
        'by_style' : (3088, 792)
    }

    # create different bin sizes
    for bin_size in [256, 128, 64, 32]:

        # operate on different problems
        for problem in ['by_country', 'by_style', 'by_product']:

            labels_train, labels_test = read_data(problem)
            images_train, images_test = read_images(problem,
            dataset_sizes[problem][0], dataset_sizes[problem][1], bin_size)

            serialize_histograms(problem, 'training', images_train, bin_size)
            serialize_histograms(problem, 'testing', images_test, bin_size)

    #train,test,labels_train,labels_test = split_data(data, labels_style)
    #train,test,labels_train,labels_test = split_data_random(data, labels_style)
    #print(labels_train)
    #model = train_svm(train, labels_train)
    #predictions = predict(model, test)
    #print(predictions)
    #print(labels_test)
    #error = calculate_error(predictions, labels_test)