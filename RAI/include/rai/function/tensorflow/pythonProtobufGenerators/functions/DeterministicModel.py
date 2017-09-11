import BaseClasses as bc
import tensorflow as tf
import core


class DeterministicModel(bc.SpecializedFunction):

    input_names = ['input']
    output_names = ['output']

    def __init__(self, dtype, gs):
        super(DeterministicModel, self).__init__(dtype, gs)

        # shortcuts
        output_dim = int(gs.output.shape[1])
        output = tf.identity(gs.output, name=self.output_names[0])
        input = gs.input

        # new placeholders
        output_target = tf.placeholder(dtype, shape=[None, output_dim], name='targetOutput')

        # solvers
        with tf.name_scope('squareLoss'):
            core.square_loss_opt(dtype, output_target, output, tf.train.AdamOptimizer)
