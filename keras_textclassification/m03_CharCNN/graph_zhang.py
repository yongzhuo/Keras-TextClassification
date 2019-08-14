# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/12 14:32
# @author   :Mo
# @function :graph of charCNN-zhang
# @paper : Character-level Convolutional Networks for Text Classiﬁcation(https://arxiv.org/pdf/1509.01626.pdf)


from __future__ import print_function, division

# char cnn
from keras.layers import Convolution1D, MaxPooling1D, ThresholdedReLU
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from keras_textclassification.base.graph import graph


class CharCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.char_cnn_layers = hyper_parameters['model'].get('char_cnn_layers',
        [[256, 7, 3], [256, 7, 3], [256, 3, -1], [256, 3, -1], [256, 3, -1], [256, 3, 3]],)
        self.full_connect_layers = hyper_parameters['model'].get('full_connect_layers', [1024, 1024],)
        self.threshold = hyper_parameters['model'].get('threshold', 1e-6)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        # x = Reshape((self.len_max, self.embed_size, 1))(embedding_output) # (None, 50, 30, 1)
        # cnn + pool
        for char_cnn_size in self.char_cnn_layers:
            x = Convolution1D(filters = char_cnn_size[0],
                              kernel_size = char_cnn_size[1],)(x)
            x = ThresholdedReLU(self.threshold)(x)
            if char_cnn_size[2] != -1:
                x = MaxPooling1D(pool_size = char_cnn_size[2],
                                 strides = 1)(x)
        x = Flatten()(x)
        # full-connect
        for full in self.full_connect_layers:
            x = Dense(units=full,)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout)(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)