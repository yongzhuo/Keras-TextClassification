# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/28 23:06
# @author   :Mo
# @function :graph of xlnet fineture, 后面不接什么网络结构, 只有一个激活层
# @paper    :XLNet: Generalized Autoregressive Pretraining for Language Understanding

from __future__ import print_function, division

from keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Dense
from keras.layers import Dropout, Reshape, Concatenate, Lambda
from keras.layers import LSTM, GRU
from keras.layers import Flatten
from keras.models import Model
from keras import backend as K
from keras import regularizers

from keras_textclassification.base.graph import graph

import numpy as np


class XlnetGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        # x = embedding_output
        x = Lambda(lambda x : x[:, -2:-1, :])(embedding_output) # 获取CLS
        # # text cnn
        # bert_output_emmbed = SpatialDropout1D(rate=self.dropout)(embedding_output)
        # concat_out = []
        # for index, filter_size in enumerate(self.filters):
        #     x = Conv1D(name='TextCNN_Conv1D_{}'.format(index),
        #                filters= self.filters_num, # int(K.int_shape(embedding_output)[-1]/self.len_max),
        #                strides=1,
        #                kernel_size=self.filters[index],
        #                padding='valid',
        #                kernel_initializer='normal',
        #                activation='relu')(bert_output_emmbed)
        #     x = GlobalMaxPooling1D(name='TextCNN_MaxPool1D_{}'.format(index))(x)
        #     concat_out.append(x)
        # x = Concatenate(axis=1)(concat_out)
        # x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output_layers = [dense_layer]
        self.model = Model(self.word_embedding.input, output_layers)
        self.model.summary(120)
