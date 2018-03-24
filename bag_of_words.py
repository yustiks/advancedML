import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os
import pathlib
import _pickle as pickle


def create_dirs(datasets_path, folder_name):

	new_paths = []

	for path in datasets_path:

		new_path = os.path.join(path, folder_name)
		pathlib.Path(new_path).mkdir(parents=True, exist_ok=True) 

		new_paths.append(new_path)

	return new_paths



def create_bow(directory, df):

	# initial data
	data_size = df.shape[0]
	bow = cv2.BOWKMeansTrainer(data_size)
	labels = df['label'].values.tolist()

	for index, row in df.iterrows():

		img_path = os.path.join(directory, row[0])

		img = cv2.imread(img_path)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		sift = cv2.xfeatures2d.SIFT_create()

		#img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		#plt.imshow(img)
		
		kp, des = sift.detectAndCompute(gray,None)

		bow.add(des)
		
		#print(len(kp), len(des))
		#cv2.imwrite('/home/dawid/sift_keypoints.jpg',des[3])

	return bow.cluster(), labels


if __name__ == "__main__":
	
	datasets_path = ('./by_style/', './by_country/', './by_product/' )
	folder_name = 'bag_of_words'
	
	new_paths = create_dirs(datasets_path, folder_name)

	print("Start processing...")
	
	for i, path in enumerate(datasets_path):

		print("Processing %s data" % path)

		for data_type in ['training', 'testing']:
			
			data = pd.read_csv(os.path.join(path, '%s.csv' % data_type), sep=',', \
				names = ["file", "label"])

			print("%s data" % data_type, data.shape, 'size')

			data_path = os.path.join(path, data_type)

			training_data, labels = create_bow(data_path, data)

			histogram_dir = os.path.join(new_paths[i], '%s_histograms_256.dat' % data_type)
			labels_dir = os.path.join(new_paths[i], '%s_labels_256.dat' % data_type)

			pickle.dump(training_data, open(histogram_dir, "wb" ))
			pickle.dump(labels, open(labels_dir, "wb" ))


