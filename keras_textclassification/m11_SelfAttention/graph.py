# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :()


from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers import SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
from keras.models import Model

from keras_textclassification.keras_layers.attention_self import AttentionSelf
from keras_textclassification.base.graph import graph


class SelfAttentionGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        x = SpatialDropout1D(self.dropout_spatial)(x)
        x = AttentionSelf(self.word_embedding.embed_size)(x)
        x_max = GlobalMaxPooling1D()(x)
        x_avg = GlobalAveragePooling1D()(x)
        x = Concatenate()([x_max, x_avg])
        x = Dropout(self.dropout)(x)
        # x = Flatten()(x)
        x = Dense(72, activation="tanh")(x)
        x = Dropout(self.dropout)(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(120)
