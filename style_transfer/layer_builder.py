import numpy as np
import tensorflow as tf


def conv_layer(X, filters, filter_size, stride, name, collection, activation=None):
    """Create a new convolution layer with Xavier initializer"""
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        
        # create Xavier initializer node 
        in_channels = int(X.get_shape()[3])
        init = tf.contrib.layers.xavier_initializer_conv2d()
    
        # create the parameter structures         
        W = tf.get_variable(initializer=init, 
                            shape=(filter_size[0], filter_size[1],
                                   in_channels, filters),
                            name="weights")
        b = tf.get_variable(initializer=tf.zeros(filters),
                            name="biases")
        
        # add parameters to the collection
        tf.add_to_collection(collection, W)
        tf.add_to_collection(collection, b)
        
        # perform convolution and add bias
        conv = tf.nn.conv2d(X, W, strides=(1, stride, stride, 1), padding="SAME")
        z = tf.nn.bias_add(conv, b)
        
        # activation function
        if activation == "relu":
            return tf.nn.relu(z)
        elif activation == "lrelu":
            return tf.nn.leaky_relu(z)
        elif activation == "sigmoid":
            return tf.nn.sigmoid(z)
        else:
            return z
     

def deconv_layer(X, filters, filter_size, stride, output_shape, name, collection, activation=None):
    """Create a new deconvolution layer with Xavier initializer"""

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        # create Xavier initializer node 
        in_channels = int(X.get_shape()[3])
        init = tf.contrib.layers.xavier_initializer_conv2d()

        # create the parameter structures         
        W = tf.get_variable(initializer=init,
                            shape=(filter_size[0], filter_size[1],
                                   filters, in_channels),
                            name="weights")
        b = tf.get_variable(initializer=tf.zeros(filters),
                            name="biases")
                
        # add parameters to the collection
        tf.add_to_collection(collection, W)
        tf.add_to_collection(collection, b)
        
        # perform convolution and add bias
        conv = tf.nn.conv2d_transpose(X, W, output_shape=output_shape,
                                      strides=(1, stride, stride, 1), padding="SAME")
        z = tf.nn.bias_add(conv, b)

        # activation function
        if activation == "relu":
            return tf.nn.relu(z)
        elif activation == "lrelu":
            return tf.nn.leaky_relu(z)
        else:
            return z
        
        
def residual_layer(X, filters, filter_size, stride, name, collection, activation=None):
    """Create a new residual double convolution layer with Xavier initializer"""
    
    # convolutions
    conv_1 = conv_layer(X, filters, filter_size, stride, name + "_conv1", collection)
    conv_2 = conv_layer(conv_1, filters, filter_size, stride, name + "_conv2", collection)
    
    # add residuals to convolution     
    z = X + conv_2

    # activation function
    if activation == "relu":
        return tf.nn.relu(z)
    elif activation == "lrelu":
        return tf.nn.leaky_relu(z)
    elif activation == "sigmoid":
        return tf.nn.sigmoid(z)
    else:
        return z