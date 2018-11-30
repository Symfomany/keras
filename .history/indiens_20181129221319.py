"""
Very Good Machine Learning Blog:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
"""


"""
    Examples
        768/768 [==============================] - 0s - loss: 0.5105 - acc: 0.7396
        Epoch 146/150
        768/768 [==============================] - 0s - loss: 0.4900 - acc: 0.7591
        Epoch 147/150
        768/768 [==============================] - 0s - loss: 0.4939 - acc: 0.7565
        Epoch 148/150
        768/768 [==============================] - 0s - loss: 0.4766 - acc: 0.7773
        Epoch 149/150
        768/768 [==============================] - 0s - loss: 0.4883 - acc: 0.7591
        Epoch 150/150
        768/768 [==============================] - 0s - loss: 0.4827 - acc: 0.7656
        32/768 [>.............................] - ETA: 0s
        acc: 78.26%
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
import os, argparse

"""
Once you have designed a network using Keras, you may want to serve it in another API, on the web, 
or other medium. One of the easiest way to do many of the above is to use the pre-built TensorFlow libraries 

In order to do this you will most likely have to ‘freeze’ your trained Keras model 
due to the way the backends of these APIs work.

Unfortunately, in order to do this easily, you will have to retrain your model in the TensorFlow implementation of Keras


Fortunately, however, this process is very simple if you know what you’re aiming for thanks to the way Keras
is integrated into TensorFlow, and various other materials provided by TensorFlow for this task.

The methodology described in this article, although written specifically for converting a model written in the Keras API 
(natively or in the TensorFlow implementation), it can also be used for any model written in native TensorFlow as well.

With TensorFlow Python API, The graph stores the information about the architecture of the network with Variable ops, whereas the checkpoint files 
contain the values of the weights at various stages of training (depending on how regularly your session checkpoints during training). 

The graph stores the information about the architecture of the network with Variable ops, 
whereas the checkpoint files *contain the values of the weights at various stages of training*

Freezing the model means producing a singular file containing information about the graph and checkpoint variables, 
but saving these hyperparameters as constants within the graph structure. 
This eliminates additional information saved in the checkpoint files such as the gradients at each point, 
which are included so that the model can be reloaded and training continued from where you left off. 

As this is not needed when serving a model purely for inference they are discarded in freezing. 
A frozen model is a file of the Google .pb file type.

In order to generate a .pb file containing the necessary information TensorFlow has helpfully written the freeze_graph.py file which, when called, will merge the TensorFlow graph and checkpoint files
The Keras API does not generate automatically this file so you will need to retrain the model in order to generate them.

There are two relatively simple ways to go about acquiring a saved trained model in the correct format for freezing. If you are using the Keras API directly, 
then you will be required to change to the Keras API implemented in a TensorFlow environment.

This should just require a change at the importing stage e.g. instead of 
from keras.layers import Convolution2D you would have from tensorflow.keras.layers import Convolution2D.


Difference between .meta files and .pbtxt files,
 well you can say .pbtxt are human readable whereas .meta files are not. 
 
 But if you keep as_test = false it is no more human readable. Still they are different . .meta files holds ,more than just the structure of the graph like MetaInfoDef , GraphDef SaverDef , CollectionDef .
  Whereas .pbtxt files holds only the structure of the graph.


"""

def freeze_graph(model_dir, output_node_names):
    """ Extrait le sous graphe défini par les noeuds de sortie et convertit
    toutes ses variables en constante 
    Args:
        model_dir: le dossier racine contenant le fichier d'état du point de contrôle
        output_node_names: une chaîne contenant tous les noms du noeud de sortie, 
                            séparées par des virgules
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

# Main Modules

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="frozen_model", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()


    # fix random seed for reproducibility
    numpy.random.seed(7)


    ## 1. Load Data

    # load pima indians dataset
    """
        Detail of csv file
        https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
    """
    dataset = numpy.loadtxt("./pima-indians-diabetes.data.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    """
    We have initialized our random number generator to ensure our results are reproducible and loaded our data.
    We are now ready to define our neural network model.
    """

    X = dataset[:,0:8]
    Y = dataset[:,8]



    ## 2. Define Model

    """
        Models in Keras are defined as a sequence of layers.
        3 layers: 3  Dense class with: Numbers of Neurones, the activation function: relu et sigmoid selon les domaines d'activation
        ReLU Vs Sigmoid: https://cdn-images-1.medium.com/max/1600/1*XxxiA0jJvPrHEJHD4z893g.png
    """

    # create Model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    """
    Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. 
    The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, 
    such as CPU or GPU or even distributed.

    Remember training a network means finding the best set of weights to make predictions for this problem.

    We must specify the loss function to use to evaluate a set of weights, 
    the optimizer used to search through different weights for the network and any optional metrics we would like to collect and report during training.

    In this case, we will use logarithmic loss, which for a binary classification problem is defined in Keras as “binary_crossentropy“. 
    We will also use the efficient gradient descent algorithm “adam” for no other reason that it is an efficient default. 


    """

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # 3. Fit the Model

    # Fit the model
    """
    batch_size: The algorithm takes the first 10 samples (from 1st to 10th) from the training dataset and trains the network.
    Apprentissage par lots avant la rétropropagation

    We can keep doing this procedure until we have propagated through all samples of the network. 
    A problem usually happens with the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder. 
    The simplest solution is just to get the final 50 samples and train the network.

    It requires less memory. 
    Typically networks train faster with mini-batches. 

    Nb of Epoq, Batch size: Nb d'entrainements 

    """
    model.fit(X, Y, epochs=10, batch_size=10)


    # 4. Evaluate Model: Accurency
    # evaluate the model

    """
    We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.
    This will only give us an idea of how well we have modeled the dataset (e.g. train accuracy), but no idea of how well the algorithm might perform on new data. 
    We have done this for simplicity, but ideally, you could separate your data into train and test datasets for training and evaluation of your model.


    Running this example, you should see a message for each of the 150 epochs printing the loss and accuracy for each,
    followed by the final evaluation of the trained model on the training dataset.


    """


    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    # 5. Make Predictions

    # round predictions
    predictions = model.predict(X)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    freeze_graph(args.model_dir, args.output_node_names)



    #  Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill. 
    #  This is a feature, not a bug. You can learn more about this in the post:


    