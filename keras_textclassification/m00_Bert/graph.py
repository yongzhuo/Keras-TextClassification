# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/13 23:06
# @author   :Mo
# @function :
from __future__ import print_function, division

from keras.engine import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


from keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Dense
from keras.layers import Dropout, Reshape, Concatenate
from keras.layers import LSTM, GRU
from keras.layers import Flatten
from keras.models import Model
from keras import backend as K
from keras import regularizers

from keras_textclassification.base.graph import graph

import numpy as np


class BertGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 650) # large, small is 300
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        # text cnn
        bert_output_emmbed = SpatialDropout1D(rate=self.dropout)(embedding_output)
        concat_out = []
        for index, filter_size in enumerate(self.filters):
            x = Conv1D(name='TextCNN_Conv1D_{}'.format(index), filters=int(K.int_shape(embedding_output)[-1]),
                       kernel_size=self.filters[index], padding='valid', kernel_initializer='normal',
                       activation='relu')(bert_output_emmbed)
            x = GlobalMaxPooling1D(name='TextCNN_MaxPool1D_{}'.format(index))(x)
            concat_out.append(x)
        x = Concatenate(axis=1)(concat_out)
        x = Dropout(self.dropout)(x)
        # x = Flatten()(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output_layers = [dense_layer]
        self.model = Model(self.word_embedding.inputs, output_layers)
        self.model.summary(120)

    def predict(self, sen):
        """
          预测，bert与其他的bert不同
        :param sen: list，input 
        :return: list, result
        """
        if type(sen)==np.ndarray:
            sen = sen.tolist()
        elif type(sen)==list:
            sen = sen
        else:
            raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
        return self.model.predict(sen)