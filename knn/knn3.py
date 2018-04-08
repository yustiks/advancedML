#!/usr/bin/env python -W ignore::DeprecationWarning
# knn on SURF and SIFT

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
#import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
#from xgboost import XGBClassifier
import warnings
from enum import Enum
import copy
#import lightgbm as lgb


class Library(Enum):
    KNN = "knn"
    SCIKIT = "scikit"
    XGBOOST = "xgboost"
    LGB = 'lgb'


def get_data(path, datasets_path):

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

			# add files paths
			for data_file in files:
				
				# get colours number in whole dataset
				# (some similar colours were connected
				# in some cases)	
				number = re.findall(r'\d+', data_file)[0]

				if not number in data_set[feature][path]:
					data_set[feature][path][number] = {
						'testing': [],
						'training': []
					}
				
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


#def plot_data(classifier, labels):

#	dot_data = tree.export_graphviz(classifier, out_file=None,  
#		class_names=labels,  
#		filled=True, rounded=True,  
#		special_characters=True
#	)  
#	graph = graphviz.Source(dot_data)  
#	graph.render("output")  


def load_initial_data():
	
	datasplit_values = ('by_style', 'by_country', 'by_product')
	label_vaues = ('decor', 'country', 'type')
	data_set = get_data('./', datasplit_values)

	# create dict
	best_values = dict()
	for data in datasplit_values:
		best_values[data] = {'accuracy': 0, 'class_names': 0, 'values': [], 'data_type': None, 'lib': None}

	# load data information
	df = pd.read_csv('./decor_extended_with_headers.csv', sep=',', header=0)


	for i, value in enumerate(datasplit_values):

		current_label = '%s_label' % label_vaues[i]

		labels_set = set(df[current_label])
		names = []

		for i_label in labels_set:

			label_name = df[df[current_label]==i_label].iloc[0][label_vaues[i]]
			names.append(label_name)

		best_values[value]['class_names'] = names

	return data_set, best_values


def get_predictions(classifiers, data_parts, colour_value):

	testing_data, testing_labels, training_data, training_labels = get_pickles(data_parts)

	scores = []

	for classifier_name, clf_object in classifiers.items():
		
		clf_object.fit(training_data, training_labels)
		score = clf_object.score(testing_data, testing_labels)
		scores.append(score)

	lib, best = (Library.KNN, 0.0)
    
	keys = list(classifiers.keys())
	for i, score in enumerate(scores):

		if score > best:
			lib, best = keys[i], score

	scores.insert(0, colour_value)

	print('Data type: %s, Scikit-Learn KNeighborsClassifier: %.2f' % tuple(scores))#print('Data type: %s, Scikit-Learn Acc: %.2f, XGBoost Acc: %.2f, LGB Acc: %.2f' % tuple(scores))

	return {
		'accuracy': best, 
		'class_names': 0, 
		'values': [testing_data, testing_labels, training_data, training_labels],
	 	'data_type': colour_value,
	 	'lib': None
 	}


if __name__ == "__main__":
	
	warnings.filterwarnings("ignore", category=DeprecationWarning)

	data_set, best_values = load_initial_data()

	empty_values = copy.deepcopy(best_values)

	rng = np.random.RandomState(1)

	# play with parameters
	classifiers = {
        Library.KNN: KNeighborsClassifier(n_neighbors=2, weights='distance'), 
		#Library.SCIKIT: tree.DecisionTreeClassifier(), 
		#Library.XGBOOST: XGBClassifier(learning_rate=0.2, n_estimators=100),
		#Library.LGB: lgb.LGBMClassifier(num_leaves=5, boosting_type='dart')
	}

	print('Loading data....\n')

	for feature, feature_data in data_set.items():
		
		print('############## Feature type:', feature)
		
		for feature_name, data in feature_data.items():

			print('Classification:', feature_name)
			
			for colour_value, data_parts in data.items(): 

				try:
						
					result = get_predictions(copy.deepcopy(classifiers), data_parts, colour_value)
					#print(result)
					if best_values[feature_name]['accuracy'] < result['accuracy']:
						
						best_values[feature_name] = result		

				except:
					continue

		print('\nBest scores:')

		for key, data in best_values.items():

			print('Feature type: %10s' % key, 'Data Type:', data['data_type'], 'Accuracy: %.2f' % data['accuracy'], 'Lib:', data['lib'])

		best_values = copy.deepcopy(empty_values)

		print('\n')

	# plot tree (works only for scikit)

	plot_dataset = best_values['by_style']

	#if plot_dataset['lib'] == Library.SCIKIT:
	#	clf_data = plot_dataset['values']
	#	tr, tr_l = np.concatenate((clf_data[0], clf_data[2])), np.concatenate((clf_data[1], clf_data[3]))

	#	clf = tree.DecisionTreeClassifier().fit(tr, tr_l)

	#	plot_data(clf, plot_dataset['class_names'])
