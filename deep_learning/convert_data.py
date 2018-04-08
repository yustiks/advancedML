import os
import cv2
import csv
import numpy as np
import tensorflow as tf


IMG_SIZE = 150
CHANNELS = 3


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf(data, labels, dataset_type):

	writing_path = os.path.join(dataset_type, 'training.tfrecords')
	writer = tf.python_io.TFRecordWriter(writing_path)

	for index in range(len(labels)):
		data_sample = data[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
		    'label': _int64_feature(int(labels[index])),
		    'image': _bytes_feature(data_sample)}))
		writer.write(example.SerializeToString())
		
	writer.close()


def process_data(dataset_type, dataset_sizes):

	reading_path = os.path.join('..', dataset_type)

	# training
	training_labels = os.path.join(reading_path, 'training.csv')
	training_data = os.path.join(reading_path, 'training')

	with open(training_labels, 'rt') as training_file:
		file_reader = csv.reader(training_file, delimiter=',')

		data = np.empty((dataset_sizes[0], IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
		labels = np.empty((dataset_sizes[0],), dtype=np.uint8)

		index = 0
		for row in file_reader:

			print('training', index)
			
			# read image as bgr
			image_bgr = cv2.imread(os.path.join(training_data, row[0]),
								   cv2.IMREAD_COLOR)

			# convert to rgb
			image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

			# add dimension
			image = image.reshape((1, IMG_SIZE, IMG_SIZE, CHANNELS))

			data[index] = image
			labels[index] = int(row[1])
			index += 1

		# convert data to tfrecords
		convert_to_tf(data, labels, 'data_' + dataset_type)


	# testing
	testing_labels = os.path.join(reading_path, 'testing.csv')
	testing_data = os.path.join(reading_path, 'testing')

	with open(testing_labels, 'rt') as testing_file:
		file_reader = csv.reader(testing_file, delimiter=',')

		data = np.empty((dataset_sizes[1], IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
		labels = np.empty((dataset_sizes[1],), dtype=np.uint8)

		index = 0
		for row in file_reader:
			
			print('testing', index)

			# read image
			image = cv2.imread(os.path.join(testing_data, row[0]),
								   cv2.IMREAD_COLOR)
			# add dimension
			image = image.reshape((1, IMG_SIZE, IMG_SIZE, CHANNELS))

			data[index] = image
			labels[index] = int(row[1])
			index += 1

		data.dump(os.path.join('data_' + dataset_type, 'testing_data.dat'))
		labels.dump(os.path.join('data_' + dataset_type, 'testing_labels.dat'))


if __name__ == '__main__':
    

	process_data('by_country', (3096, 784))
	process_data('by_product', (3096, 784))
	process_data('by_style', (3088, 792))