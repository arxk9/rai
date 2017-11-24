import BaseClasses as bc
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


# multiple gated recurrent unit layers (https://arxiv.org/pdf/1406.1078v3.pdf)
class GRUNet(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):
        super(GRUNet, self).__init__(dtype)
        nonlin_str = param[0]
        nonlin = getattr(tf.nn, nonlin_str)

        dimension = [int(i) for i in param[1:]]
        self.input = tf.placeholder(dtype, shape=[None, None, dimension[0]], name=fn.input_names[0])  # [batch, time, dim]
        length_ = tf.placeholder(dtype, shape=[None], name='length')  # [batch]
        length_ = tf.cast(length_, dtype=tf.int32)
        self.seq_length = tf.reshape(length_, [-1])

        cells = []
        state_size = []
        recurrent_state_size = 0

        for size in dimension[1:]:
            cell = rnn.GRUCell(size, activation=nonlin, kernel_initializer=tf.contrib.layers.xavier_initializer())
            cells.append(cell)
            recurrent_state_size += cell.state_size
            state_size.append(cell.state_size)

        cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        hiddenStateDim = tf.identity(tf.constant(value=[recurrent_state_size], dtype=tf.int32), name='h_dim')

        init_state = tf.placeholder(dtype=dtype, shape=[None, recurrent_state_size], name='h_init')
        init_state_tuple = tuple(tf.split(init_state, num_or_size_splits=state_size, axis=1))

        # Full-length output for training
        self.output, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.input, sequence_length=self.seq_length, dtype=dtype, initial_state=init_state_tuple)

        hiddenState = tf.concat([state for state in final_state], axis=1, name='h_state')

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
        self.net = None
