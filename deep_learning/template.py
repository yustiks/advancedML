import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from read_data import *


#----------[ HYPER-PARAMETERS ]----------#
EPOCHS = 1
BATCH_SIZE = 8
#----------------------------------------#


with tf.device('/cpu:0'):
    
    # read data from the feeding queues
    images, labels = inputs('by_style', BATCH_SIZE, EPOCHS)
        
    # Normalize the images     
    images = (tf.cast(images, tf.float32) / 255.0)

	#################################################################
	#!!! build the graph here by adding operations to the images !!!#
	#################################################################


# create the initializers for all tensors in the graph 
init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

# create a session for running operations in the graph
sess = tf.Session()

# initialize the variables
sess.run(init_op)

# start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# this is the main loop of the training session
# it reads data on and on until the specified number of epochs is reached
try:
    
	step = 0
	while not coord.should_stop():

		# run training steps or whatever
		step += 1
		print("Step " + str(step), flush=True)

		###########################################################################
		#!!! evaluate variables from the graph here by adding them to sess.run !!!#
		###########################################################################

		img, label = sess.run([images, labels])
		plt.imshow(img[0])
		plt.title("Label " + str(label[0]))
		plt.show()

        
except tf.errors.OutOfRangeError:
    
    print('\nDone training -- epoch limit reached\n')
    
finally:
    
    # When done, ask the threads to stop.
    coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()