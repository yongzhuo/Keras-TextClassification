# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of CRNN
# @paper    :A C-LSTM Neural Network for Text Classification(https://arxiv.org/abs/1511.08630)


from keras import regularizers
from keras.models import Model
from keras.layers import SpatialDropout1D, Conv1D
from keras.layers import Dropout, Flatten, Dense, Concatenate
from keras.layers import LSTM, GRU, Bidirectional, CuDNNLSTM, CuDNNGRU

from keras_textclassification.base.graph import graph


class CRNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 650)  # large, small is 300
        self.dropout_spatial = hyper_parameters['model'].get('dropout_spatial', 0.2)
        self.l2 = hyper_parameters['model'].get('l2', 0.001)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        embedding_output_spatial = SpatialDropout1D(self.dropout_spatial)(x)

        if self.rnn_units=="LSTM":
                layer_cell = LSTM
        elif self.rnn_units=="GRU":
                layer_cell = GRU
        elif self.rnn_units=="CuDNNLSTM":
                layer_cell = CuDNNLSTM
        elif self.rnn_units=="CuDNNGRU":
                layer_cell = CuDNNGRU
        else:
            layer_cell = GRU
        # CNN
        convs = []
        for kernel_size in self.filters:
            conv = Conv1D(self.filters_num,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='SAME',
                            kernel_regularizer=regularizers.l2(self.l2),
                            bias_regularizer=regularizers.l2(self.l2),
                            )(embedding_output_spatial)
            convs.append(conv)
        x = Concatenate(axis=1)(convs)
        # Bi-LSTM, 论文中使用的是LSTM
        x = Bidirectional(layer_cell(units=self.rnn_units,
                                     return_sequences=True,
                                     activation='relu',
                                     kernel_regularizer=regularizers.l2(self.l2),
                                     recurrent_regularizer=regularizers.l2(self.l2)
                                     ))(x)
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(120)
