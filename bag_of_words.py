import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os
import pathlib
import _pickle as pickle
from enum import Enum


class Algorithm(Enum):

	SWIFT = "SWIFT"
	SURF = "SURF"


def create_dirs(datasets_path, folder_name):

	new_paths = []

	for path in datasets_path:

		new_path = os.path.join(path, folder_name)
		pathlib.Path(new_path).mkdir(parents=True, exist_ok=True) 

		new_paths.append(new_path)

	return new_paths



def create_bow(directory, df, algorithm=Algorithm.SWIFT, value=None):

	if value is None:
		value = 0 if algorithm == Algorithm.SWIFT else 100

	# initial data
	data_size = df.shape[0]
	bow = cv2.BOWKMeansTrainer(data_size)
	labels = df['label'].values.tolist()

	i = 0

	for index, row in df.iterrows():

		file_name = row[0]

		if int(os.path.splitext(file_name)[0]) % 7 != 0:
			continue

		if (int(os.path.splitext(file_name)[0]) / 7) % 10 != 0:
			continue

		img_path = os.path.join(directory, file_name)

		img = cv2.imread(img_path)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		if algorithm == Algorithm.SWIFT:
			sift = cv2.xfeatures2d.SIFT_create(70)
			kp, des = sift.detectAndCompute(gray, None)

		else:
			surf = cv2.xfeatures2d.SURF_create(value)
			kp, des = surf.detectAndCompute(gray, None)
		
		

		bow.add(des)

		if not i%10:
			print(i, 'files done')
		
		i += 1



		#img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		#plt.imshow(img)
		#print(len(kp), len(des))
		#cv2.imwrite('/home/dawid/sift_keypoints.jpg',des[3])

	return bow.cluster(), labels


if __name__ == "__main__":
	
	datasets_path = ('./by_style/', './by_country/', './by_product/' )
	folder_names = ['swift_bag_of_words', 'surf_bag_of_words']
	new_paths = []

	for folder in folder_names:
		new_paths.append(create_dirs(datasets_path, folder))

	print("Start processing...")

	for j, folder in enumerate(folder_names):

		print('Processing ', folder)

		algorithm = Algorithm.SWIFT if j == 0 else Algorithm.SURF
	
		for i, path in enumerate(datasets_path):

			print("Processing %s data" % path)

			for data_type in ['training', 'testing']:

				current_path = new_paths[j][i]
				
				data = pd.read_csv(os.path.join(path, '%s.csv' % data_type), sep=',', \
					names = ["file", "label"])

				print("%s data" % data_type, data.shape, 'size')

				data_path = os.path.join(path, data_type)

				training_data, labels = create_bow(data_path, data, algorithm)

				histogram_dir = os.path.join(current_path, '%s_histograms_256.dat' % data_type)
				labels_dir = os.path.join(current_path, '%s_labels_256.dat' % data_type)

				pickle.dump(training_data, open(histogram_dir, "wb" ))
				pickle.dump(labels, open(labels_dir, "wb" ))


