import os
import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 150
CHANNELS = 3


def read_and_decode(filename_queue):

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
	  serialized_example,
	  features={
	      'image': tf.FixedLenFeature([], tf.string),
	      'label': tf.FixedLenFeature([], tf.int64),
	  })

	image = tf.decode_raw(features['image'], tf.uint8)
	image.set_shape((IMG_SIZE * IMG_SIZE * CHANNELS))
	image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, CHANNELS))
	label = tf.cast(features['label'], tf.int32)

	return image, label


def inputs(dataset_type, batch_size, num_epochs):

	filename = os.path.join('data_' + dataset_type, 'training.tfrecords')

	with tf.name_scope('input'):
		
		filename_queue = tf.train.string_input_producer(
		    [filename], num_epochs=num_epochs)

		image, label = read_and_decode(filename_queue)

		images, sparse_labels = tf.train.shuffle_batch(
		    [image, label], batch_size=batch_size, num_threads=2,
		    capacity=10000 + 3 * batch_size,
		    min_after_dequeue=10000)

		return images, sparse_labels