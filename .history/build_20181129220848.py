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



def freeze_graph(model_dir, output_node_names):
      """Extract the sub graph defined by the output nodes and convert 
  all its variables into constant 
  Args:
      model_dir: the root folder containing the checkpoint state file
      output_node_names: a string, containing all the output node's names, 
                          comma separated
                        """
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
    
  # We precise the file fullname of our freezed graph
  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  output_graph = absolute_model_dir + "/frozen_model.pb"

  # We clear devices to allow TensorFlow to control on which device it will load operations
  clear_devices = True

  # We start a session using a temporary fresh Graph
  with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
      output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

  return output_graph_def

freeze_graph('./TF_Keras/',"output_node/Softmax")

