import BaseClasses as bc
import tensorflow as tf
import core


class Vfunction(bc.SpecializedFunction):
    input_names = ['state']
    output_names = ['value']

    def __init__(self, dtype, gs):
        super(Vfunction, self).__init__(dtype, gs)

        # variables
        state_dim = gs.input.shape[1]
        state = gs.input
        value = tf.identity(gs.output, name=self.output_names[0])
        clip_param = tf.Variable(0.2, name='clip_param')
        max_grad_norm = tf.Variable(0.5, name='max_grad_norm')

        # new placeholders
        value_target = tf.placeholder(dtype, shape=[None, 1], name='targetValue')
        value_pred = tf.placeholder(dtype, shape=[None, 1], name='predictedValue')

        tf.identity(value_pred, name='test')

        # Assign ops.
        param_assign_placeholder = tf.placeholder(dtype, shape=[1, 1], name='param_assign_placeholder')
        tf.assign(clip_param, tf.reshape(param_assign_placeholder, []), name='clip_param_assign')
        tf.assign(max_grad_norm, tf.reshape(param_assign_placeholder, []), name='grad_param_assign')

        # gradients
        jac_V_wrt_State = tf.identity(tf.gradients(tf.reduce_mean(value), state)[0], name='gradient_AvgOf_V_wrt_State')

        # solvers
        with tf.name_scope('trainUsingTargetValue'):
            learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            core.square_loss_opt(dtype, value_target, value, tf.train.AdamOptimizer(learning_rate=learning_rate), maxnorm=max_grad_norm)

        with tf.name_scope('trainUsingTargetValue_huber'):
            learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            core.huber_loss_opt(dtype, value_target, value, tf.train.AdamOptimizer(learning_rate=learning_rate), maxnorm=max_grad_norm)

        with tf.name_scope('trainUsingTRValue'):
            # Clipping-based trust region loss (https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py)
            vpredclipped = value_pred + tf.clip_by_value(value - value_pred, tf.negative(clip_param), clip_param)
            vfloss1 = tf.square(value - value_target)
            vfloss2 = tf.square(vpredclipped - value_target)
            TR_loss = .5 * tf.reduce_mean(tf.maximum(vfloss1, vfloss2), name='loss')

            #Optimize
            learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grad = tf.gradients(TR_loss, gs.l_param_list, colocate_gradients_with_ops=True)
            grads, gradnorm = tf.clip_by_global_norm(grad, clip_norm=max_grad_norm)
            grads = zip(grads, gs.l_param_list)
            train = optimizer.apply_gradients(grads, name='solver')