import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = '.'
PATH_TO_SPRITE_IMAGE = 'sprite.png'
PATH_TO_DATA = 'features_embeddings_16.dat'

# Create randomly initialized embedding weights which will be trained.
data = np.load(PATH_TO_DATA)
embedding_var = tf.Variable(data, name='img_embedding')

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = 'labels_016.tsv'


embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
# Specify the width and height of a single thumbnail.
embedding.sprite.single_image_dim.extend([150, 150])


# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

# Start a session
with tf.Session() as sess:
	sess.run(init_op)

	step = 0
	saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)