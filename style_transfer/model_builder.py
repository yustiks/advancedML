import numpy as np
import tensorflow as tf

from layer_builder import *


def generator(input_img, name):

	name = "GEN_" + name
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

		conv_1 = conv_layer(input_img, 32, (7, 7), 1, "conv_1", name, "lrelu")
		conv_2 = conv_layer(conv_1, 64, (3, 3), 2, "conv_2", name, "lrelu")
		conv_3 = conv_layer(conv_2, 128, (3, 3), 2, "conv_3", name, "lrelu")

		res_1 = residual_layer(conv_3, 128, (3, 3), 1, "res_1", name, "lrelu")
		res_2 = residual_layer(res_1, 128, (3, 3), 1, "res_2", name, "lrelu")
		res_3 = residual_layer(res_2, 128, (3, 3), 1, "res_3", name, "lrelu")

		batch_size = int(res_3.get_shape()[0])

		deconv_2 = deconv_layer(res_3, 64, (3, 3), 2,
		 			[batch_size, 75, 75, 64], "deconv_2", name, "lrelu")
		deconv_1 = deconv_layer(deconv_2, 32, (3, 3), 2,
					[batch_size, 150, 150, 32], "deconv_1", name, "lrelu")
		G = deconv_layer(deconv_1, 3, (7, 7), 1,
					[batch_size, 150, 150, 3], "G", name, "relu")

		return G


def discriminator(input_img, name):

	name = "DSC_" + name
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

		conv_1 = conv_layer(input_img, 32, (3, 3), 1, "conv_1", name, "lrelu")
		conv_2 = conv_layer(conv_1, 32, (3, 3), 1, "conv_2", name, "lrelu")
		conv_3 = conv_layer(conv_2, 32, (3, 3), 1, "conv_3", name, "lrelu")
		conv_4 = conv_layer(conv_3, 1, (5, 5), 5, "conv_4", name, "sigmoid")
		pred = tf.reduce_mean(tf.contrib.layers.flatten(conv_4), -1, keep_dims=False, name="D")

		return pred