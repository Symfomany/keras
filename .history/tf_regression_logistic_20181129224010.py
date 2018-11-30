import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, argparse

"""
Any interaction with your filesystem to save persistent data in TF needs a Saver object and a Session object.

The Saver constructor allows you to control many things among which 1 is important:

The var_list: Default to None, this is the list of variables you want to persist to your filesystem.
You can either choose to save all the variables, some variables or even a dictionary to give custom names to your variables.

The Session constructor allows you to control 3 things:

+ The var_list: This is used in case of a distributed architecture to handle computation. You can specify which TF server or ‘target’ you want to compute on.
+ The graph: the graph you want the Session to handle. The tricky thing for beginners is the fact that there is always a default Graph in TF where all operations are set by default, so you are always in a “default Graph scope”.
+ The config: You can use ConfigProto to configure TF. Check the linked source for more details.

The Saver can handle the saving and loading (called restoring) of your Graph metadata and your Variables data. 
To do that, it adds operations inside the current Graph that will be evaluated within a session.

By default, the Saver will handle the default Graph and all its included Variables, 
but you can create as much Savers as you want to control any graph or subgraph and their variables.

If you look at your folder, it actually creates 3 files per save call and a checkpoint file,
 I’ll go into more details about this in the annexe.
 
You can go on just by understanding that weights are saved into .data files and your graph
and metadata are saved into the .meta file.

Note: You must be careful to use a Saver with a Session linked to the Graph containing all the variables the Saver is handling.😨

To restore a meta checkpoint, use the TF helper import_meta_graph:

import tensorflow as tf

# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('results/model.ckpt-1000.meta')

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()

# Finally we can retrieve tensors, operations, collections, etc.
global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
train_op = graph.get_operation_by_name('loss/train_op')
hyperparameters = tf.get_collection('hyperparameters')

"""

dir = os.path.dirname(os.path.realpath(__file__))

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
def get_dataset():
    """
        Method used to generate the dataset
    """
    # Numbers of row per class
    row_per_class = 100
    # Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick_2, healthy, healthy_2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    targets = targets.reshape(-1, 1)

    return features, targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="tf_models", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

 
    features, targets = get_dataset()
    # Plot points
    #plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    #plt.show()

    tf_features = tf.placeholder(tf.float32, shape=[None, 2])
    tf_targets = tf.placeholder(tf.float32, shape=[None, 1])

    # First
    w1 = tf.Variable(tf.random_normal([2, 3]))
    b1 = tf.Variable(tf.zeros([3]))
    # Operations
    z1 = tf.matmul(tf_features, w1) + b1
    a1 = tf.nn.sigmoid(z1)

    # Output neuron
    w2 = tf.Variable(tf.random_normal([3, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    # Operations
    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)

    cost = tf.reduce_mean(tf.square(py - tf_targets))

    correct_prediction = tf.equal(tf.round(py), tf_targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for e in range(100):

        sess.run(train, feed_dict={
            tf_features: features,
            tf_targets: targets
        })

        print("accuracy =", sess.run(accuracy, feed_dict={
            tf_features: features,
            tf_targets: targets
        }))

    # We can check easily that we are indeed in the default graph
    print(z1.graph == tf.get_default_graph())


    # By default, the Saver handles every Variables related to the default graph
    all_saver = tf.train.Saver() 

    all_saver.save(sess, args.model_dir + '/data')
    # freeze_graph(args.model_dir, args.output_node_names)
