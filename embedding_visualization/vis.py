import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = '.'

# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 128 # Dimensionality of the embedding.
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
# embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

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