import BaseClasses as bc
import tensorflow as tf
import core


class RecurrentVfunction(bc.SpecializedFunction):
    input_names = ['state']
    output_names = ['value']

    def __init__(self, dtype, gs):
        super(RecurrentVfunction, self).__init__(dtype, gs)
        # variables
        value = tf.squeeze(gs.output, axis=2, name=self.output_names[0])

        state_dim = gs.input.shape[1]
        state = gs.input

        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param')
        max_grad_norm = tf.Variable(tf.constant(0.5, dtype=dtype), name='max_grad_norm')

        # new placeholders
        value_target = tf.placeholder(dtype, shape=[None, None], name='targetValue')
        value_pred = tf.placeholder(dtype, shape=[None, None], name='predictedValue')
        mask = tf.sequence_mask(gs.seq_length, name='mask')
        value_target_masked = tf.boolean_mask(value_target, mask)
        value_pred_masked = tf.boolean_mask(value_pred, mask)
        value_masked = tf.boolean_mask(value, mask)

        # Assign ops.
        param_assign_placeholder = tf.placeholder(dtype, shape=[1, 1], name='param_assign_placeholder')
        tf.assign(clip_param, tf.reshape(param_assign_placeholder, []), name='clip_param_assign')
        tf.assign(max_grad_norm, tf.reshape(param_assign_placeholder, []), name='grad_param_assign')

        # solvers
        with tf.name_scope('trainUsingTargetValue'):
            core.square_loss_opt(dtype, value_target_masked, value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=max_grad_norm)

        with tf.name_scope('trainUsingTargetValue_huber'):
            core.huber_loss_opt(dtype, value_target_masked, value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=max_grad_norm)

        with tf.name_scope('trainUsingTRValue'):
            # Clipping-based trust region loss (https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py)
            # core.square_loss_opt(dtype, value_target, value, tf.train.AdamOptimizer)

            vfloss1 = tf.square(value_masked - value_target_masked)
            clip_rate = clip_param[0]

            vpredclipped = value_pred_masked + tf.clip_by_value(value_masked - value_pred_masked , -clip_rate, clip_rate)
            vfloss2 = tf.square(vpredclipped - value_target_masked)

            TR_loss = .5 * tf.reduce_mean(tf.maximum(vfloss1, vfloss2), name='loss')

            #Optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            grad = tf.gradients(TR_loss, gs.l_param_list, colocate_gradients_with_ops=True)
            grads, gradnorm = tf.clip_by_global_norm(grad, clip_norm=max_grad_norm)
            grads = zip(grads, gs.l_param_list)
            train = optimizer.apply_gradients(grads, name='solver', global_step=tf.train.get_global_step())
