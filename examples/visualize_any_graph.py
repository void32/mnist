"""
Run the script: 
$ python visualize_any_graph.py

then start the tensorbord server
$ tensorboard --logdir=output

and visit the webpagefor the local tensorbord at http://localhost:6006


You can simply create a session just to write the graph to the FileWriter and not do anything else
See https://stackoverflow.com/questions/48391075/is-it-possible-to-visualize-a-tensorflow-graph-without-a-training-op

Troubleshooting:
Use global_variables_initializer to initialize all variables, see https://stackoverflow.com/questions/36007883/tensorflow-attempting-to-use-uninitialized-value-in-variable-initialization
    init = tf.initialize_all_variables()
    sess.run(init)
"""

import tensorflow as tf

a = tf.add(1, 2, name="Add_these_numbers")
b = tf.multiply(a, 3)
c = tf.add(4, 5, name="And_These_ones")
d = tf.multiply(c, 6, name="Multiply_these_numbers")
e = tf.multiply(4, 5, name="B_add")
f = tf.div(c, 6, name="B_mul")
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    sess_run_out = sess.run(h)
    print(sess_run_out)
    writer.close()
