import os
import sys
import cv2
import csv
import pathlib
import numpy as np
import tensorflow as tf


def get_styles_images(filename):
	"""
	Return a dictionary of images by style
	"""
	
	with open(os.path.join('..', filename), 'rt', newline='') \
		as reading_csv:
		data_reader = csv.reader(reading_csv, delimiter=',', quotechar='|')

		# ignore the header
		header = next(data_reader)

		# dictionary to hold image names for each style
		style_images = {}

		for data_line in data_reader:

			# unpack values
			style = data_line[3].replace(' ', '_')
			img_name = data_line[6]

			# append image to its corresponding style
			style_images.setdefault(style, []).append(img_name)

		return style_images


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf(data, data_path):

	writing_path = os.path.join(data_path, 'training.tfrecords')
	writer = tf.python_io.TFRecordWriter(writing_path)

	for index in range(len(data)):
		data_sample = data[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
		    'image': _bytes_feature(data_sample)}))
		writer.write(example.SerializeToString())
		
	writer.close()


def serialize_style(images_path, style_element, test_samples_no, samples_per_img):

	# unpack the elements
	style = style_element[0]
	img_names = style_element[1]

	# create output directory
	try:
		style_path = os.path.join('data', style)
		pathlib.Path(style_path).mkdir(parents=True, exist_ok=True)
		print('\t' + style, flush=True)
	except:
		raise Exception('Cannot create the data directory ' + styles)

	# read images from file
	images = np.asarray(list(map((lambda img_name: 
		cv2.cvtColor(
			(cv2.imread(os.path.join('..', images_path, img_name), cv2.IMREAD_COLOR)),
			cv2.COLOR_BGR2RGB)),
		img_names)))

	# split the data into training and test
	training_data = images[:-(test_samples_no * samples_per_img)]
	testing_data = images[-(test_samples_no * samples_per_img):]

	# serialize training
	convert_to_tf(training_data, style_path)

	# serialize test
	testing_data.dump(os.path.join(style_path, 'test.dat'))



if __name__ == "__main__":

	print('Creating styles dataset...')

	# create data directory
	pathlib.Path('data').mkdir(parents=True, exist_ok=True)

	styles_dict = get_styles_images('decor_extended.csv')
	for style_elem in styles_dict.items():
		serialize_style('images_extended', style_elem, 4, 8)
	
	print('Styles dataset is ready!')
