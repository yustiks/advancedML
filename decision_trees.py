import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os
import glob
import re
import _pickle as pickle
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz


def get_data(path, datasets_path):

	colur_values = ["{0:0>3}".format(2**x) for x in range(5, 9)]

	# create dataset
	data_set = dict()		

	# path to get all posible features 
	# any path can be used (from by_style etc...)
	# cause all should have the same features
	example_path = os.path.join('./', datasets_path[0], '*')

	feature_paths = [os.path.basename(os.path.normpath(path)) for path in glob.glob(example_path + '[!testing][!training]*/')]

	for feature in feature_paths:

		data_set[feature] = dict()

		# get data files for each dataset
		for path in datasets_path:

			data_set[feature][path] = dict()
			
			# curent data directory
			current_path = os.path.join('./', path, feature)

			# files from current data directory
			files = glob.glob("%s/*.dat" % current_path)

			# organize files by colours
			for colour_val in colur_values:

				data_set[feature][path][colour_val] = {
					'testing': [],
					'training': []
				}

			# add files paths
			for data_file in files:
				
				# get colours number in whole dataset
				# (some similar colours were connected
				# in some cases)	
				number = re.findall(r'\d+', data_file)[0]

				# split testing and training dataset
				data_type = 'testing'

				if not data_type in data_file:
					data_type = 'training'
				
				data_set[feature][path][number][data_type].append(data_file)

	return data_set


def get_pickles(sets):

	pickles = [0, 0, 0, 0]
	
	for key, paths in sets.items():

		for path in paths:
			
			index = 0 if key == 'testing' else 2
			
			if 'labels' in path:
				index +=1 

			pickles[index] = pickle.load(open(path, "rb" ))

	return pickles


def plot_data(classifier):
	dot_data = tree.export_graphviz(classifier, out_file=None, 
		feature_names=iris.feature_names,  
		class_names=iris.target_names,  
		filled=True, rounded=True,  
		special_characters=True
	)  
	graph = graphviz.Source(dot_data)  
	graph 

if __name__ == "__main__":
	
	datasplit_values = ('by_style', 'by_country', 'by_product')
	data_set = get_data('./', datasplit_values)

	best_accuracy = 0
	best_values = dict.fromkeys(datasplit_values, {'accuracy': 0, 'class_names': 0})

	# load data information
	df = pd.read_csv('./decor.csv', sep=',', header=0)

	style_labels, country_labels, product_labels = set(df['decor_label']), set(df['country_label']), set(df['type_label'])

	print(country_labels, style_labels, product_labels)
	for label in country_labels:
		print(label, df[df.country>='1'].head(1))
	sys.exit(2)

	print('Loading data....')

	for feature, feature_data in data_set.items():
		
		print('Feature type:', feature)
		
		for feature_name, data in feature_data.items():

			print('Classification:', feature_name)
			
			for colour_value, data_parts in data.items(): 

				testing_data, testing_labels, training_data, training_labels = get_pickles(data_parts)

				classifier = tree.DecisionTreeClassifier().fit(training_data, training_labels)

				result = classifier.predict(testing_data)

				accuracy = accuracy_score(testing_labels, result)

				if accuracy > best_accuracy:
					best_values.append(testing_data, testing_labels, training_data, training_labels)

				print('Data type: %s, Accuracy: %.2f' % (colour_value, accuracy))


