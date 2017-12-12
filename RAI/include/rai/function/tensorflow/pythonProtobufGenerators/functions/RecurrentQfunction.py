import BaseClasses as bc
import tensorflow as tf
import core


class RecurrentQfunction(bc.SpecializedFunction):
    input_names = ['state', 'sampledAction']
    output_names = ['QValue']

    def __init__(self, dtype, gs):
        super(RecurrentQfunction, self).__init__(dtype, gs)
        # variables
        q_value = tf.squeeze(gs.output, axis=2, name=self.output_names[0])

        state = gs.input1
        action = gs.input2

        # variables
        max_grad_norm = tf.Variable(tf.constant(0.5, dtype=dtype), name='max_grad_norm')

        # new placeholders
        q_value_target = tf.placeholder(dtype, shape=[None, None], name='targetQValue') #[batch, time, 1]
        mask = tf.sequence_mask(gs.seq_length, name='mask')
        q_value_target_masked = tf.boolean_mask(q_value_target, mask)
        q_value_masked = tf.boolean_mask(q_value, mask)
        avg = tf.reduce_mean(q_value_masked, name='average_Q_value')

        # Assign Ops
        param_assign_placeholder = tf.placeholder(dtype, [1,1], name='gradNorm_placeholder')
        gradNorm_assign = tf.assign(max_grad_norm, tf.reshape(param_assign_placeholder, []), name='max_norm_assign')

        # gradients
        jac_Q_wrt_State = tf.identity(tf.gradients(avg, gs.input1)[0], name='gradient_AvgOf_Q_wrt_State')
        jac_Q_wrt_Action = tf.identity(tf.gradients(avg, gs.input2)[0], name='gradient_AvgOf_Q_wrt_action')

        # solvers
        with tf.name_scope('trainUsingTargetQValue'):
            core.square_loss_opt(dtype, q_value_target_masked, q_value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=max_grad_norm)

        with tf.name_scope('trainUsingTargetQValue_huber'):
            core.huber_loss_opt(dtype, q_value_target_masked, q_value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=max_grad_norm)
