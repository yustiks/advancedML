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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


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


def plot_data(classifier, labels):

	dot_data = tree.export_graphviz(classifier, out_file=None,  
		class_names=labels,  
		filled=True, rounded=True,  
		special_characters=True
	)  
	graph = graphviz.Source(dot_data)  
	graph.render("output")  


def load_initial_data():
	
	datasplit_values = ('by_style', 'by_country', 'by_product')
	label_vaues = ('decor', 'country', 'type')
	data_set = get_data('./', datasplit_values)

	# create dict
	best_values = dict()
	for data in datasplit_values:
		best_values[data] = {'accuracy': 0, 'class_names': 0, 'values': [], 'data_type': None}

	# load data information
	df = pd.read_csv('./decor.csv', sep=',', header=0)


	for i, value in enumerate(datasplit_values):

		current_label = '%s_label' % label_vaues[i]

		labels_set = set(df[current_label])
		names = []

		for i_label in labels_set:

			label_name = df[df[current_label]==i_label].iloc[0][label_vaues[i]]
			print(current_label, label_name, value)
			names.append(label_name)

		best_values[value]['class_names'] = names

	return data_set, best_values

if __name__ == "__main__":
	
	data_set, best_values = load_initial_data()
	
	rng = np.random.RandomState(1)

	print('Loading data....')

	for feature, feature_data in data_set.items():
		
		print('Feature type:', feature)
		
		for feature_name, data in feature_data.items():

			print('Classification:', feature_name)
			
			for colour_value, data_parts in data.items(): 

				testing_data, testing_labels, training_data, training_labels = get_pickles(data_parts)

				classifier = tree.DecisionTreeClassifier().fit(training_data, training_labels)

				accuracy = classifier.score(testing_data, testing_labels)

				if best_values[feature_name]['accuracy'] < accuracy:
					
					best_values[feature_name]['accuracy'] = accuracy
					best_values[feature_name]['values'] = [testing_data, testing_labels, training_data, training_labels]
					best_values[feature_name]['data_type'] = colour_value

				print('Data type: %s, Accuracy: %.2f' % (colour_value, accuracy))


	print()
	print('Best scores:')

	for key, data in best_values.items():

		print('Feature type:', key, 'Data Type:', data['data_type'], 'Accuracy: %.2f' % data['accuracy'])


	plot_dataset = best_values['by_style']
	clf_data = plot_dataset['values']
	tr, tr_l = np.concatenate((clf_data[0], clf_data[2])), np.concatenate((clf_data[1], clf_data[3]))

	clf = tree.DecisionTreeClassifier().fit(tr, tr_l)

	plot_data(clf, plot_dataset['class_names'])