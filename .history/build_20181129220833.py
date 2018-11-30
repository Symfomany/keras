"""
First, we build the model in the Keras API implemented in TensorFlow, 
be sure to name your input and output layers, as we will need these names later:
"""

#import keras modules in tensorflow implementation
from tensorflow import keras
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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



#define input layer
inpt = Input(shape = (28,28,1), name = "input_node")

#call the model
logits = CNN(inpt)

#define model
model = Model(inpt,logits)

#compile the model
model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001), \
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

#convert to an Estimator
# the model_dir states where the graph and checkpoint files will be saved to
estimator_model = tf.keras.estimator.model_to_estimator(keras_model = model, \
                                                        model_dir = './models')


"""
input_function: features random shuffle batch and epoches
"""
def input_function(features,labels=None,shuffle=False):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_node": features},
        y=labels,
        shuffle=shuffle,
        batch_size = 5,
        num_epochs = 1
    )
    return input_fn
  
estimator_model.train(input_fn = input_function(X_train,y_train,True))



Keras_MNIST
