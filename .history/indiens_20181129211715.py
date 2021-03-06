"""
Very Good Machine Learning Blog:
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy
import os, argparse

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
model.fit(X, Y, epochs=150, batch_size=10)


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

"""

...
    Epoch 145/150
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


#  Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill. 
#  This is a feature, not a bug. You can learn more about this in the post:


def  freeze_graph ( model_dir , output_node_names ):
    """ Extrait le sous graphe défini par les noeuds de sortie et convertit
    toutes ses variables en constante 
    Args:
        model_dir: le dossier racine contenant le fichier d'état du point de contrôle
        output_node_names: une chaîne contenant tous les noms du noeud de sortie, 
                            séparées par des virgules
    """
    si  non tf.gfile.Exists (model_dir):
        raise  AssertionError (
            " Le répertoire d'exportation n'existe pas. Veuillez spécifier une exportation "
            " répertoire: % s "  % model_dir)

    si  non output_node_names:
        print ( " Vous devez fournir le nom d'un noeud à --output_node_names. " )
        retour  - 1

    # Nous récupérons notre checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state (rép_abus)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # Nous précisons le nom complet du fichier de notre graphe gelé
    absolute_model_dir =  " / " .join (input_checkpoint.split ( ' / ' ) [: - 1 ])
    output_graph = absolute_model_dir +  " /frozen_model.pb "

    # Nous autorisons les périphériques à permettre à TensorFlow de contrôler sur quel périphérique il chargera les opérations
    clear_devices =  True

    # Nous commençons une session en utilisant un nouveau graphique temporaire
    avec tf.Session ( graph = tf.Graph ()) as sess:
        # Nous importons le méta-graphe dans le graphe par défaut actuel
        saver = tf.train.import_meta_graph (input_checkpoint +  ' .meta ' , clear_devices = clear_devices)

        # Nous restaurons les poids
        saver.restore (sess, input_checkpoint)

        # Nous utilisons un assistant TF intégré pour exporter des variables en constantes
        output_graph_def = tf.graph_util.convert_variables_to_constants (
            sess, # La session est utilisée pour récupérer les poids
            tf.get_default_graph (). as_graph_def (), # Le graph_def est utilisé pour récupérer les nœuds
            output_node_names.split ( " , " ) # Les noms des noeuds de sortie sont utilisés pour sélectionner les noeuds utiles.
        ) 

        # Enfin, nous sérialisons et exportons le graphique de sortie dans le système de fichiers
        avec tf.gfile.GFile (output_graph, " wb " ) en tant que f:
            f.write (output_graph_def.SerializeToString ())
        print ( " % d ops dans le graphique final. "  %  len (output_graph_def.node))

    return output_graph_def
