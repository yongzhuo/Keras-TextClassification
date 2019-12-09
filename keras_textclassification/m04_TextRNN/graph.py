# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of base


from keras.layers import LSTM, GRU, Bidirectional, CuDNNLSTM, CuDNNGRU
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.models import Model

from keras_textclassification.base.graph import graph


class TextRNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.num_rnn_layers = hyper_parameters['model'].get('num_rnn_layers', 2)
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 256)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        # x = Reshape((self.len_max, self.embed_size, 1))(embedding)
        if self.rnn_type=="LSTM":
                layer_cell = LSTM
        elif self.rnn_type=="GRU":
                layer_cell = GRU
        elif self.rnn_type=="CuDNNLSTM":
                layer_cell = CuDNNLSTM
        elif self.rnn_type=="CuDNNGRU":
                layer_cell = CuDNNGRU
        else:
            layer_cell = GRU

        # Bi-LSTM
        for nrl in range(self.num_rnn_layers):
            x = Bidirectional(layer_cell(units=self.rnn_units,
                                         return_sequences=True,
                                         activation='relu',
                                         kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                         recurrent_regularizer=regularizers.l2(0.32)
                                         ))(x)
            x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(120)
