# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 11:45
# @author   :Mo
# @function :char CNN of 'Yoon Kim'
# paper: 2015, Character-Aware Neural Language Models(https://arxiv.org/abs/1508.06615)

from __future__ import print_function, division

import keras
import numpy as np
from keras import backend as K
from keras import regularizers
from keras.engine import Layer
from keras.initializers import Constant
from keras.layers import Bidirectional, GRU
# char cnn
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Multiply, Add, Lambda
from keras.layers import Dropout, Reshape, Concatenate, BatchNormalization
from keras.layers import TimeDistributed, Flatten
from keras.models import Model

from keras_textclassification.base.graph import graph


class CharCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.char_cnn_layers = hyper_parameters['model'].get('char_cnn_layers',
            [[50, 1], [100, 2], [150, 3], [200, 4], [200, 5], [200, 6], [200, 7]]) # large
            # [[25, 1], [50, 2], [75, 3], [100, 4], [125, 5], [150, 6]])  # small
        self.highway_layers = hyper_parameters['model'].get('highway_layers', 2)
        self.num_rnn_layers = hyper_parameters['model'].get('num_rnn_layers', 2)
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 650) # large, small is 300
        self.len_max_word = hyper_parameters['model'].get('len_max_word', 30)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        embedding_output = Reshape((self.len_max, self.embed_size, 1))(embedding_output) # (None, 50, 30, 1)
        embedding_output = Concatenate()([embedding_output for i in range(self.len_max_word)]) # (None, 50, 30, 21)
        embedding_output = Reshape((self.len_max, self.len_max_word, self.embed_size))(embedding_output) # (None, 50, 21, 30)
        conv_out = []
        for char_cnn_size in self.char_cnn_layers:
            conv = Convolution2D(name='Convolution2D_{}_{}'.format(char_cnn_size[0], char_cnn_size[1]),
                                 filters=char_cnn_size[0],
                                 kernel_size= (1, char_cnn_size[1]),
                                 activation='tanh')(embedding_output)
            pooled = MaxPooling2D(name='MaxPooling2D_{}_{}'.format(char_cnn_size[0], char_cnn_size[1]),
                                  pool_size=(1, self.len_max_word - char_cnn_size[1] + 1)
                                  )(conv)
            conv_out.append(pooled)
        x = Concatenate()(conv_out) #  (None, 50, 1, 1100)
        x = Reshape((self.len_max, K.int_shape(x)[2] * sum(np.array([ccl[0] for ccl in self.char_cnn_layers]))))(x) # (None, 50, 1100)
        x = BatchNormalization()(x)
        # Highway layers
        for hl in range(self.highway_layers):
            # 两个都可以，第二个是我自己写的
            # x = TimeDistributed(Highway(activation='sigmoid', transform_gate_bias=-2, input_shape=K.int_shape(x)[1:2]))(x)
            x = TimeDistributed(Lambda(highway_keras, name="highway_keras"))(x)
        # rnn layers
        for nrl in range(self.num_rnn_layers):
            x = Bidirectional(GRU(units=self.rnn_units, return_sequences=True,
                                         kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                         recurrent_regularizer=regularizers.l2(0.32)
                                   ))(x)
            # x = GRU(units=self.rnn_units, return_sequences=True,
            #                        kernel_regularizer=regularizers.l2(0.32 * 0.1),
            #                        recurrent_regularizer=regularizers.l2(0.32)
            #                        )(x)

            x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)


def highway_keras(x):
    # writter by my own
    # paper； Highway Network(http://arxiv.org/abs/1505.00387).
    # 公式
    # 1. s = sigmoid(Wx + b)
    # 2. z = s * relu(Wx + b) + (1 - s) * x
    # x shape : [N * time_depth, sum(filters)]

    # Table 1. CIFAR-10 test set accuracy of convolutional highway networks with
    # rectified linear activation and sigmoid gates.
    # For comparison, results reported by Romero et al. (2014)
    # using maxout networks are also shown.
    # Fitnets were trained using a two step training procedure using soft targets from the trained Teacher network,
    # which was trained using backpropagation. We trained all highway networks directly using backpropagation.
    # * indicates networks which were trained only on a set of 40K out of 50K examples in the training set.



    # Figure 2. Visualization of certain internals of the blocks in the best 50 hidden layer highway networks trained on MNIST
    # (top row) and CIFAR-100 (bottom row). The first hidden layer is a plain layer which changes the dimensionality of the representation to 50. Each of
    # the 49 highway layers (y-axis) consists of 50 blocks (x-axis).
    # The first column shows the transform gate biases, which were initialized to -2 and -4 respectively.
    # In the second column the mean output of the transform gate over 10,000 training examples is depicted.
    # The third and forth columns show the output of the transform gates and
    # the block outputs for a single random training sample.

    gate_transform = Dense(units=K.int_shape(x)[1],
                           activation='sigmoid',
                           use_bias=True,
                           kernel_initializer='glorot_uniform',
                           bias_initializer=keras.initializers.Constant(value=-2))(x)
    gate_cross = 1 - gate_transform
    block_state = Dense(units=K.int_shape(x)[1],
                        activation='relu',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zero')(x)
    high_way = gate_transform * block_state + gate_cross * x

    return high_way

gg = 0
class Highway(Layer):
    """
      codes from github: https://github.com/batikim09/Keras_highways/blob/master/src/conv2d_highway.py
    """
    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-2, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        self.dense_1 = Dense(units=dim, bias_initializer=Constant(self.transform_gate_bias))
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=dim)
        self.dense_2.build(input_shape)
        self.trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights
        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config

# def highway(input_, size, num_layers=1, bias=-2.0, f=K.relu):
#     """ github : https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py
#     Highway Network (cf. http://arxiv.org/abs/1505.00387).
#     t = sigmoid(Wy + b)
#     z = t * g(Wy + b) + (1 - t) * y
#     where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
#     """
#
#     def linear(input_, output_size, scope=None):
#         '''
#         Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
#         Args:
#             args: a tensor or a list of 2D, batch x n, Tensors.
#         output_size: int, second dimension of W[i].
#         scope: VariableScope for the created subgraph; defaults to "Linear".
#       Returns:
#         A 2D Tensor with shape [batch x output_size] equal to
#         sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
#       Raises:
#         ValueError: if some of the arguments has unspecified or wrong shape.
#       '''
#
#         shape = input_.get_shape().as_list()
#         if len(shape) != 2:
#             raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
#         if not shape[1]:
#             raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
#         input_size = shape[1]
#
#         # Now the computation.
#         matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
#         bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
#
#         return K.matmul(input_, tf.transpose(matrix)) + bias_term
#
#     for idx in range(num_layers):
#         g = f(linear(input_, size, scope='highway_lin_%d' % idx))
#
#         t = K.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
#
#         output = t * g + (1. - t) * input_
#         input_ = output
#     return output

gg = 0

# def highway_network(embedding, units):
#         # github: https://github.com/SeonbeomKim/TensorFlow-lstm-char-cnn/blob/master/lstm_char_cnn.py
# 		# embedding: [N*time_depth, sum(filters)]
# 		transform_gate = tf.layers.dense(
# 				embedding,
# 				units=units,
# 				activation=tf.nn.sigmoid,
# 				kernel_initializer=self.initializer,
# 				bias_initializer=tf.constant_initializer(-2)
# 			) # [N*time_depth, sum(filters)]
#
# 		carry_gate = 1-transform_gate # [N*time_depth, sum(filters)]
# 		block_state = tf.layers.dense(
# 				embedding,
# 				units=units,
# 				activation=tf.nn.relu,
# 				kernel_initializer=self.initializer,
# 				bias_initializer=self.initializer
# 			) # [N*time_depth, sum(filters)]
# 		highway = transform_gate * block_state + carry_gate * embedding # [N*time_depth, sum(filters)]
# 			# if transfor_gate is 1. then carry_gate is 0. so only use block_state
# 			# if transfor_gate is 0. then carry_gate is 1. so only use embedding
# 			# if transfor_gate is 0.@@. then carry_gate is 0.@@. so use sum of scaled block_state and embedding
# 		return highway # [N*time_depth, sum(filters)]

gg = 0
