import numpy as np
import cv2
import os
import pandas as pd

    
"""read data from .csv file"""
def read_data(problem):
    path = os.getcwd()
    path_train = os.path.join(path, problem, 'training.csv')
    path_test = os.path.join(path, problem, 'testing.csv')
    train = pd.read_csv(path_train, encoding = "Latin-1")
    test = pd.read_csv(path_test, encoding = "Latin-1")
    labels_train = train.iloc[:,1]
    labels_test = test.iloc[:,1]
    return np.asarray(labels_train[::8]), np.asarray(labels_test[::8])


"""read images from file"""
def read_images(problem, training_size, testing_size, clusters_size):

    images_train = np.zeros((training_size, 3 * clusters_size))
    images_test = np.zeros((testing_size, 3 * clusters_size))
    im_counter = 0

    for image_name in os.listdir(os.path.join(problem, 'training')):
        
        if int(image_name[:4]) % 8 > 0:
            continue

        print('{0} - {1} - training - {2}'.format(
        	problem, clusters_size, image_name))
        train = os.path.join(problem, 'training', image_name)
        # print(train)
        row = each_image(train, clusters_size)
        images_train[im_counter, :] = row                           
        im_counter += 1
    im_counter = 0

    for image_name in os.listdir(os.path.join(problem, 'testing')):
        
        if int(image_name[:4]) % 8 > 0:
            continue

        print('{0} - {1} - testing - {2}'.format(
        	problem, clusters_size, image_name))
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

    hsv_img = cv2.cvtColor(recomposed_img, cv2.COLOR_RGB2HSV)
    b,g,r = cv2.split(hsv_img)
    data[0, 0:clusters_size] = make_histogram(r, clusters_size)
    data[0, clusters_size:2*clusters_size] = make_histogram(g, clusters_size)
    data[0, 2*clusters_size:] = make_histogram(b, clusters_size)

    # normalize histogram
    return data * (3 * clusters_size / sum(data[0]))


"""make histograms"""
def make_histogram(img, clusters_size):
    # apply mask so that only the patterns and interiors are taken into account
    hist, bins = np.histogram(img, bins=clusters_size)
    hist = hist.reshape(-1,1)
    hist = hist.transpose()
    return hist


def serialize_histograms(problem, dataset_type, dataset, labels, clusters_size):
	
	path = os.path.join(problem, dataset_type + \
	 ('_hsv_histograms_%.3d.dat' % clusters_size))
	dataset.dump(path)

	labels_path = os.path.join(problem, dataset_type + \
	('_labels_%.3d.dat' % clusters_size))
	labels.dump(labels_path)
	

if __name__ == "__main__":  
    
    dataset_sizes = {
        'by_country' : (387, 98),
        'by_product' : (387, 98),
        'by_style' : (386, 99)
    }

    # create different bin sizes
    for bin_size in [16, 10]:

        # operate on different problems
        for problem in ['by_style']:
            labels_train, labels_test = read_data(problem)
            images_train, images_test = read_images(problem,
            dataset_sizes[problem][0], dataset_sizes[problem][1], bin_size)

            serialize_histograms(problem, 'training', images_train, labels_train, bin_size)
            serialize_histograms(problem, 'testing', images_test, labels_test, bin_size)