import tensorflow as tf
import numpy as np

with tf.Session() as session:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.mul(a, b, name="c")

    initialize_all_variables_op = tf.initialize_variables(tf.all_variables(), name='initializeAllVariables')
    summary_writer = tf.train.SummaryWriter('/tmp/debug_tf', graph_def=session.graph_def)

    with tf.name_scope('learnableParameters'):
        tf.identity(tf.constant(value=0, dtype=tf.int32), name='numberOf')

    tf.train.write_graph(session.graph_def, '', 'simple_graph.pb', as_text=False)



