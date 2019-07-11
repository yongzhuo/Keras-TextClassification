# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of base


from keras.layers import Reshape, Concatenate, Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from keras_textclassification.base.graph import graph


class TextCNNGraph(graph):
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
        embedding = self.word_embedding.output
        embedding_reshape = Reshape((self.len_max, self.embed_size, 1))(embedding)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for filter in self.filters:
            conv = Conv2D(filters = self.filters_num,
                          kernel_size = (filter, self.embed_size),
                          padding = 'valid',
                          kernel_initializer = 'normal',
                          activation = 'relu',
                          )(embedding_reshape)
            pooled = MaxPool2D(pool_size = (self.len_max - filter + 1, 1),
                               strides = (1, 1),
                               padding = 'valid',
                               )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = Concatenate(axis=1)(conv_pools)
        x = Flatten()(x)
        x = Dropout(self.dropout)(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)
