import functions.Policy as pc
import tensorflow as tf
import core
import Utils
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import backend as K
from operator import mul
from functools import reduce
import numpy as np


class RecurrentStochasticPolicy(pc.Policy):
    def __init__(self, dtype, gs):
        # shortcuts
        action_dim = int(gs.output.shape[-1])
        state_dim = int(gs.input.shape[-1])

        action = tf.identity(gs.output, name=self.output_names[0])

        # standard deviation layer
        with tf.name_scope('stdevconcatOutput'):
            wo = tf.Variable(tf.zeros(shape=[1, action_dim], dtype=dtype), name='W',
                             trainable=True)  # Log standard deviation
            action_stdev = tf.identity(tf.exp(wo), name='stdev')

        gs.l_param_list.append(wo)
        gs.a_param_list.append(wo)

        super(RecurrentStochasticPolicy, self).__init__(dtype, gs)

        stdev_assign_placeholder = tf.placeholder(dtype, shape=[1, action_dim], name='Stdev_placeholder')
        Stdev_assign = tf.assign(wo, tf.log(stdev_assign_placeholder), name='assignStdev')
        Stdev_get = tf.exp(wo, name='getStdev')

        tangent_in = tf.placeholder(dtype, shape=[1, None], name='tangent')
        old_stdv = tf.placeholder(dtype, shape=[1, action_dim], name='stdv_o')
        old_action_sampled = tf.placeholder(dtype, shape=[None, None, action_dim], name='sampled_oa')
        old_action_noise = tf.placeholder(dtype, shape=[None, None, action_dim], name='noise_oa')
        advantage_in = tf.placeholder(dtype, shape=[None, None], name='advantage')


        # Algorithm params
        kl_coeff = tf.Variable(tf.ones(dtype=dtype, shape=[1, 1]), name='kl_coeff')
        ent_coeff = tf.Variable(0.01 * tf.ones(dtype=dtype, shape=[1, 1]), name='ent_coeff')
        clip_param = tf.Variable(0.2 * tf.ones(dtype=dtype, shape=[1, 1]), name='clip_param')
        PPO_params_placeholder = tf.placeholder(dtype=dtype, shape=[1, 3], name='PPO_params_placeholder')

        param_assign_op_list = []
        param_assign_op_list += [tf.assign(kl_coeff, tf.slice(PPO_params_placeholder, [0, 0], [1, 1]), name='kl_coeff_assign')]
        param_assign_op_list += [tf.assign(ent_coeff, tf.slice(PPO_params_placeholder, [0, 1], [1, 1]), name='ent_coeff_assign')]
        param_assign_op_list += [tf.assign(clip_param, tf.slice(PPO_params_placeholder, [0, 2], [1, 1]), name='clip_param_assign')]

        PPO_param_assign_ops = tf.group(*param_assign_op_list, name='PPO_param_assign_ops')

        with tf.name_scope('trainUsingGrad'):
            gradient_from_critic = tf.placeholder(dtype, shape=[1, None], name='Inputgradient')
            train_using_critic_learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'),
                                                          shape=[])
            train_using_grad_optimizer = tf.train.AdamOptimizer(learning_rate=train_using_critic_learning_rate)

            split_parameter_gradients = tf.split(gradient_from_critic,
                                                 [reduce(mul, param.get_shape().as_list(), 1) for param in
                                                  gs.a_param_list], 1)
            manipulated_parameter_gradients = []
            for grad, param in zip(split_parameter_gradients, gs.l_param_list):
                manipulated_parameter_gradients += [tf.reshape(grad, tf.shape(param))]

            manipulated_parameter_gradients_and_parameters = zip(manipulated_parameter_gradients, gs.l_param_list)
            train_using_gradients = train_using_grad_optimizer.apply_gradients(
                manipulated_parameter_gradients_and_parameters, name='applyGradients')

        util = Utils.Utils(dtype)

        with tf.name_scope('Algo'):
            mask = tf.sequence_mask(gs.seq_length, name='mask')
            logp_n = tf.boolean_mask(util.log_likelihood(action, action_stdev, old_action_sampled), mask)
            logp_old = tf.boolean_mask(util.log_likelihood(old_action_noise, old_stdv), mask)
            advantage = tf.boolean_mask(advantage_in, mask)
            ratio = tf.exp(logp_n - logp_old)
            ent = tf.reduce_sum(wo + .5 * tf.cast(tf.log(2.0 * np.pi * np.e), dtype=dtype), axis=-1)
            meanent = tf.reduce_mean(ent)

            with tf.name_scope('PPO'):
                surr1 = tf.multiply(ratio, advantage)
                clip_rate = clip_param[0]
                surr2 = tf.multiply(tf.clip_by_value(ratio, 1.0 - clip_rate, 1.0 + clip_rate), advantage)
                PPO_loss = tf.reduce_mean(tf.maximum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

                kl_ = tf.boolean_mask(
                    util.kl_divergence((old_action_sampled - old_action_noise), old_stdv, action, action_stdev), mask)
                kl_mean = tf.reshape(tf.reduce_mean(kl_), shape=[1, 1, 1], name='kl_mean')

                Total_loss = PPO_loss - tf.multiply(ent_coeff, meanent)
                Total_loss2 = PPO_loss - tf.multiply(ent_coeff, meanent) + tf.multiply(kl_coeff, tf.reduce_mean(kl_))

                policy_gradient = tf.identity(tf.expand_dims(util.flatgrad(Total_loss, gs.l_param_list), axis=0), name='Pg')  # flatgrad

                policy_gradient2 = tf.identity(tf.expand_dims(util.flatgrad(Total_loss2, gs.l_param_list), axis=0), name='Pg2')  # flatgrad

                #
                # meanfixed = tf.stop_gradient(action)
                # stdfixed = tf.stop_gradient(action_stdev)
                # kl_ = tf.reduce_mean(util.kl_divergence(meanfixed, stdfixed, action, action_stdev), name='kld')
                #
                # print(kl_)
                #
                # dkl_dth = tf.identity(tf.gradients(kl_, gs.l_param_list[0]))  # flatgrad
                # print(dkl_dth)
                # print(gs.l_param_list)
                #
                # l_param_assign_split = [tf.ones(shape = tf.shape(param)) for param in
                #                                   gs.l_param_list]
                #
                # lp_assign_op_list = []
                #
                # for idx, param in enumerate(gs.l_param_list):
                #     param.assign(l_param_assign_split[idx])
                #     print(param)
                #

                # flat = tf.reshape(gs.l_param_list[0], [-1])
                # print(flat)
                # gradc = tf.gradients(dkl_dth[0], gs.l_param_list[0])
                # print("???")
                # print(gradc)
                #
                # gvp = tf.reduce_sum(tf.multiply(dkl_dth, tangent_input))
                # print(gvp)
                # #
                # fvp = tf.identity(util.flatgrad(gvp, gs.l_param_list), name='fvp')  # flatgrad, super slow
                # print(fvp)

                # def getfvp(tangent):
                #     temp = tf.reduce_sum(tf.multiply(dkl_dth, tangent))
                #     return util.flatgrad(temp, gs.l_param_list)
                #
                # out1, out2 = util.CG_tf(getfvp, tangent_input, 100, 1e-15)
                # Ng = tf.identity(out1, name='Cg')
                # err = tf.identity(out2, name='Cgerror')



                # for i in range(0, len(gs.rnn_layers)):
                #     # TODO : apply return_state in r1.3~
                #     states.append(gs.rnn_layers[i](state)[:, -1, :])
                #
                # h_state = tf.concat([state for state in states], axis=1)
                # h_state = tf.expand_dims(h_state, axis=2, name='h_state')
                # print(h_state)

                # print(gs.states)
                # print( tf.reshape(gs.states[0][1], [-1]))
                # h_size = tf.concat([tf.reshape(state[1], [-1]) for state in gs.states], axis=1, name='h_size')
                # print(h_size)
                # h_state = tf.identity(gs.states,name='h_states')
