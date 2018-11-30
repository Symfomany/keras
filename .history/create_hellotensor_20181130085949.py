# Create a simple TF Graph 
# By Omid Alemi - Jan 2017
# Works with TF <r1.0

import tensorflow as tf

I = tf.placeholder(tf.float32, shape=[None,3], name='I') # input
W = tf.Variable(tf.zeros_initializer(shape=[3,2]), dtype=tf.float32, name='W') # weights
b = tf.Variable(tf.zeros_initializer(shape=[2]), dtype=tf.float32, name='b') # biases
O = tf.nn.relu(tf.matmul(I, W) + b, name='O') # activation / output

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  
  # save the graph
  tf.train.write_graph(sess.graph_def, '.', 'hellotensor.pbtxt')  

  # normally you would do some training here
  # we will just assign something to W
  sess.run(tf.assign(W, [[1, 2],[4,5],[7,8]]))
  sess.run(tf.assign(b, [1,1]))

  #save a checkpoint file, which will store the above assignment  
  saver.save(sess, 'hellotensor.ckpt')
  