import os
import cv2
import csv
import pathlib
import numpy as np
from random import shuffle
from augment_images import remove_noisy_lines, augment_image


def serialize_samples(data_dict, criteria, read_directory='./images'):
	"""
		Serialize samples and store them in appropriate directory
	"""

	# create output directories if they don't exist
	try:
		pathlib.Path(criteria).mkdir(parents=True, exist_ok=True)
		pathlib.Path(os.path.join(
			criteria, 'training')).mkdir(parents=True, exist_ok=True) 
		pathlib.Path(os.path.join(
			criteria, 'testing')).mkdir(parents=True, exist_ok=True) 
	except:
		raise Exception('Cannot create the data directories!')

	# serialize training images
	for purpose in ['training', 'testing']:

		with open(os.path.join(criteria, purpose + '.csv'), 'wt', newline='') as writing_csv:
			data_writer = csv.writer(writing_csv, delimiter=',', quotechar='|')

			index = 0
			write_directory = os.path.join(criteria, purpose)
			for img_name, img_label in data_dict[purpose]:
				# read the image
				image = cv2.imread(os.path.join(read_directory, img_name),
								   cv2.IMREAD_COLOR)
				# augment the image
				for new_img in augment_image(image):
					# serialize the image
					new_img_name = '%.4d.png' % index
					cv2.imwrite(os.path.join(write_directory, new_img_name), new_img)
					# update the csv
					data_writer.writerow([new_img_name, img_label])
					index += 1


def divide_by_country(data_lines, training_percentage=0.8):
	"""
		Divide samples by country to ensure balance in the training/testing sets
	"""

	training_samples = []
	test_samples = []
	countries_size = 4

	for c in range(1, countries_size + 1):
		# select only one class samples at a time
		data_subset = [(d[0], d[1]) for d in data_lines if d[1] == c]
		# shuffle the list
		shuffle(data_subset)
		# append examples to the training set
		training_samples += data_subset[:int(training_percentage * len(data_subset))]
		test_samples += data_subset[int(training_percentage * len(data_subset)):]
		
	data_dict = {
		'training' : training_samples,
		'testing' : test_samples
	}
	serialize_samples(data_dict, 'by_country')


def divide_by_style(data_lines, training_percentage=0.8):
	"""
		Divide samples by style to ensure balance in the training/testing sets
	"""

	training_samples = []
	test_samples = []
	styles_size = 7

	for c in range(1, styles_size + 1):
		# select only one class samples at a time
		data_subset = [(d[0], d[2]) for d in data_lines if d[2] == c]
		# shuffle the list
		shuffle(data_subset)
		# append examples to the training set
		training_samples += data_subset[:int(training_percentage * len(data_subset))]
		test_samples += data_subset[int(training_percentage * len(data_subset)):]
		
	data_dict = {
		'training' : training_samples,
		'testing' : test_samples
	}
	serialize_samples(data_dict, 'by_style')


def divide_by_product(data_lines, training_percentage=0.8):
	"""
		Divide samples by style to ensure balance in the training/testing sets
	"""
	
	training_samples = []
	test_samples = []
	product_size = 2

	for c in range(1, product_size + 1):
		# select only one class samples at a time
		data_subset = [(d[0], d[3]) for d in data_lines if d[3] == c]
		# shuffle the list
		shuffle(data_subset)
		# append examples to the training set
		training_samples += data_subset[:int(training_percentage * len(data_subset))]
		test_samples += data_subset[int(training_percentage * len(data_subset)):]
		
	data_dict = {
		'training' : training_samples,
		'testing' : test_samples
	}
	serialize_samples(data_dict, 'by_product')


def divide_by_country(data_lines, training_percentage=0.8):
	"""
		Divide samples by country to ensure balance in the training/testing sets
	"""

	training_samples = []
	test_samples = []
	countries_size = 4

	for c in range(1, countries_size + 1):
		# select only one class samples at a time
		data_subset = [(d[0], d[1]) for d in data_lines if d[1] == c]
		# shuffle the list
		shuffle(data_subset)
		# append examples to the training set
		training_samples += data_subset[:int(training_percentage * len(data_subset))]
		test_samples += data_subset[int(training_percentage * len(data_subset)):]
		
	data_dict = {
		'training' : training_samples,
		'testing' : test_samples
	}
	serialize_samples(data_dict, 'by_country')



if __name__ == "__main__":

	data_size = 485
	data_lines = data_size * [None]

	# open csv for reading data
	with open('decor.csv', 'rt', newline='') as reading_csv:
		data_reader = csv.reader(reading_csv, delimiter=',', quotechar='|')

		# ignore the header
		header = next(data_reader)

		for index, data_line in enumerate(data_reader):
			# extract only the numerical info from the data samples
			data_lines[index] = [data_line[-1]] + \
			                    [int(data_line[i]) for i in [0, 2, 4]]

	print('Dividing by country...', flush=True)
	divide_by_country(data_lines)
	print('Dividing by style...', flush=True)
	divide_by_style(data_lines)
	print('Dividing by product...', flush=True)
	divide_by_product(data_lines)