"""
First, we build the model in the Keras API implemented in TensorFlow, 
be sure to name your input and output layers, as we will need these names later:
"""

#import keras modules in tensorflow implementation
from tensorflow import keras
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense

import numpy as np
import tensorflow as tf

def CNN(input_layer):
  '''
  defines the layers of the model
  input_layer - pass input layer name
  '''
  conv1 = Convolution2D(16, 2, padding = 'same', activation = 'relu')(input_layer)
  pool1 = MaxPool2D(pool_size = 2)(conv1)
  
  conv2 = Convolution2D(32, 2, padding = 'same', activation = 'relu')(pool1)
  pool2 = MaxPool2D(pool_size = 2)(conv2)
    
  flat = Flatten()(pool2)
  dense = Dense(128, activation = 'relu')(flat)
    
  output = Dense(10, activation  = 'softmax', name = "output_node")(dense)
  return output
