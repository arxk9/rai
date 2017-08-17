import tensorflow as tf
import numpy as np

with tf.Session() as session:

    input_dim = 1
    output_dim = 1

    hidden_sizes = (300, 400)

    dtype = tf.float32


    input = tf.placeholder(dtype, shape=[None, input_dim], name='input')

    top = input

    with tf.name_scope('hiddenLayer1'):
        W1 = tf.Variable(tf.random_uniform(shape=[input_dim, hidden_sizes[0]], minval=-1/np.sqrt(input_dim), maxval=1/np.sqrt(input_dim)), name='W')
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_sizes[0]]), name='b')
        top = tf.nn.relu(tf.matmul(top, W1) + b1)

    with tf.name_scope('hiddenLayer2'):
        W2 = tf.Variable(tf.random_uniform(shape=[hidden_sizes[0], hidden_sizes[1]], minval=-1/np.sqrt(hidden_sizes[0]), maxval=1/np.sqrt(hidden_sizes[0])), name='W')
        b2 = tf.Variable(tf.constant(shape=[hidden_sizes[1]], value=0.1), name='b')
        top = tf.nn.relu(tf.matmul(top, W2) + b2)

    with tf.name_scope('outputLayer'):
        Wo = tf.Variable(tf.random_uniform(shape=[hidden_sizes[1], output_dim], minval=-3e-3, maxval=3e-3), name='W')
        bo = tf.Variable(tf.constant(shape=[output_dim], value=0.1), name='b')
        top = tf.matmul(top, Wo) + bo

    output = tf.identity(top, name='output')

    parameters = [W1, b1, W2, b2, Wo, bo]

    assign_op_list = []
    interpolate_op_list = []

    with tf.name_scope('learnableParameters'):
        tau = tf.placeholder(dtype, name='tau')
        tf.identity(tf.constant(value=len(parameters), dtype=tf.int32), name='numberOf')
        for idx, parameter in enumerate(parameters):
            tf.identity(tf.constant(value=parameter.name, dtype=tf.string), name='name_%d'%idx)

            parameter_assign_placeholder = tf.placeholder(dtype, name='parameterPlaceholder_%d'%idx)
            assign_op_list += [parameter.assign(parameter_assign_placeholder)]
            interpolate_op_list += [parameter.assign(parameter_assign_placeholder*tau + parameter*(1-tau))]

        learnable_parameters_assign_all = tf.group(*assign_op_list, name='assignAll')
        learnable_parameters_interpolate_all = tf.group(*interpolate_op_list, name='interpolateAll')


    output_target = tf.placeholder(dtype, shape=[None, output_dim], name='targetOutput')
    loss = tf.reduce_mean(tf.square(output - output_target))
    regularization_term = tf.add_n([1e-4 * tf.nn.l2_loss(var) for var in parameters])
    regularized_loss = loss + regularization_term

    with tf.name_scope('trainUsingTargetOutput'):
        solver_learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
        solver_op = tf.train.AdamOptimizer(learning_rate=solver_learning_rate).minimize(regularized_loss, var_list=parameters, name='solver')

    initialize_all_variables_op = tf.initialize_variables(tf.all_variables(), name='initializeAllVariables')
    summary_writer = tf.train.SummaryWriter('/tmp/debug_tf', graph_def=session.graph_def)

    tf.train.write_graph(session.graph_def, '', 'simple_mlp_with_transferable_weights.pb', as_text=False)

