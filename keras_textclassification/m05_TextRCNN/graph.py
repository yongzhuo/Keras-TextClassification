# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 11:45
# @author   :Mo
# @function :RCNN model
# paper: Recurrent Convolutional Neural Networks for TextClassiﬁcation(http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)


from __future__ import print_function, division

from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Dense, Lambda
from keras.layers import Dropout, Reshape, Concatenate
from keras.layers import LSTM, GRU
from keras.layers import Flatten
from keras.models import Model
from keras import backend as K
from keras import regularizers

from keras_textclassification.base.graph import graph



class RCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 256) # large, small is 300
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络，行卷积加池化
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        # rnn layers
        if self.rnn_units=="LSTM":
                layer_cell = LSTM
        else:
            layer_cell = GRU
        # 反向
        x_backwords = layer_cell(units=self.rnn_units,
                                    return_sequences=True,
                                    kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                    recurrent_regularizer=regularizers.l2(0.32),
                                    go_backwards = True)(embedding_output)
        x_backwords_reverse = Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
        # 前向
        x_fordwords = layer_cell(units=self.rnn_units,
                                    return_sequences=True,
                                    kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                    recurrent_regularizer=regularizers.l2(0.32),
                                    go_backwards = False)(embedding_output)
        # 拼接
        x_feb = Concatenate(axis=2)([x_fordwords, embedding_output, x_backwords_reverse])

        ####使用多个卷积核##################################################
        x_feb = Dropout(self.dropout)(x_feb)
        # Concatenate后的embedding_size
        dim_2 = K.int_shape(x_feb)[2]
        x_feb_reshape = Reshape((self.len_max, dim_2, 1))(x_feb)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for filter in self.filters:
            conv = Conv2D(filters = self.filters_num,
                          kernel_size = (filter, dim_2),
                          padding = 'valid',
                          kernel_initializer = 'normal',
                          activation = 'relu',
                          )(x_feb_reshape)
            pooled = MaxPooling2D(pool_size = (self.len_max - filter + 1, 1),
                                   strides = (1, 1),
                                   padding = 'valid',
                                   )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = Concatenate()(conv_pools)
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        #########################################################################
        x = Dense(units=128, activation="tanh")(x)
        x = Dropout(self.dropout)(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def create_model_cls(self, hyper_parameters):
        """
            构建神经网络, col, 论文中maxpooling使用的是列池化, 不过实验效果似乎不佳，而且训练速度超级慢
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        # rnn layers
        if self.rnn_units=="LSTM":
                layer_cell = LSTM
        else:
            layer_cell = GRU
        # 反向
        x_backwords = layer_cell(units=self.rnn_units,
                                    return_sequences=True,
                                    kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                    recurrent_regularizer=regularizers.l2(0.32),
                                    go_backwards = True)(embedding_output)
        x_backwords_reverse = Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
        # 前向
        x_fordwords = layer_cell(units=self.rnn_units,
                                    return_sequences=True,
                                    kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                    recurrent_regularizer=regularizers.l2(0.32),
                                    go_backwards = False)(embedding_output)
        # 拼接
        x_feb = Concatenate(axis=2)([x_fordwords, embedding_output, x_backwords_reverse])

        ####列池化##################################################
        x_feb = Dropout(self.dropout)(x_feb)
        dim_2 = K.int_shape(x_feb)[2]
        x_feb_reshape = Reshape((dim_2, self.len_max))(x_feb)

        conv_pools = []
        for filter in self.filters:
            conv = Conv1D(filters=self.filters_num, # filter=300
                          kernel_size=filter,
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          )(x_feb_reshape)
            pooled = MaxPooling1D(padding='valid',
                                  pool_size=32,
                                  )(conv)
            conv_pools.append(pooled)
        x = Concatenate(axis=1)(conv_pools)
        # x = MaxPooling1D(padding = 'VALID',)(x_feb_reshape)
        x = Flatten()(x)
        x = Dropout(self.dropout)(x)

        #########################################################################

        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)



# 卷积的2种方式
# # 1 github: https://github.com/ShawnyXiao/TextClassification-Keras/tree/master/model/RCNN/rcnn.py
# x = Conv1D(64, kernel_size=1, activation='tanh')(x)
# x = GlobalMaxPooling1D()(x)
#
#
# # 2 github : https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier/blob/master/recurrent_convolutional_keras.py
# semantic = Conv1D(hidden_dim_2, kernel_size=1, activation="tanh")()  # See equation (4).
# # Keras provides its own max-pooling layers, but they cannot handle variable length input
# # (as far as I can tell). As a result, I define my own max-pooling layer here.
# pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (5).



