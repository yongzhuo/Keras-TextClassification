# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :Hierarchical Attention Networks for Document Classification(https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)


from keras.layers import Dense, Dropout, SpatialDropout1D, Flatten, Input
from keras.layers import Bidirectional, LSTM, GRU, TimeDistributed
from keras import regularizers
from keras.models import Model
import keras.backend as K

from keras_textclassification.keras_layers.Attention_self import AttentionSelf
from keras_textclassification.base.graph import graph


class HANGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'Bidirectional-LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 256)
        self.attention_units = hyper_parameters['model'].get('attention_units', self.rnn_units*2)
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        self.len_max_sen = hyper_parameters['model'].get('len_max_sen', 50)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        # char or word
        x_input_word = self.word_embedding.output
        x_word = self.word_level()(x_input_word)
        x_word_to_sen = AttentionSelf(self.attention_units)(x_word)
        x_word_to_sen = Dropout(self.dropout)(x_word_to_sen)

        # sentence or doc
        x_sen = self.sentence_level()(x_word_to_sen)
        x_sen = AttentionSelf(self.attention_units)(x_sen)
        x_sen = Dropout(self.dropout)(x_sen)

        x_sen = Flatten()(x_sen)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x_sen)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(120)

    def word_level(self):
        x_input_word = Input(shape=(None, self.len_max))
        x = SpatialDropout1D(self.dropout_spatial)(x_input_word)
        x = Bidirectional(GRU(units=self.rnn_units,
                              return_sequences=True,
                              activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2),
                              recurrent_regularizer=regularizers.l2(self.l2)))(x)
        out_sent = AttentionSelf(2 * self.rnn_units)(x)
        model = Model(x_input_word, out_sent)
        return model

    def sentence_level(self):
        x_input_sen = Input(shape=(self.len_max_sen, 2 * self.rnn_units))
        x = SpatialDropout1D(self.dropout_spatial)(x_input_sen)
        output_doc = Bidirectional(GRU(units=2 * self.rnn_units,
                              return_sequences=True,
                              activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2),
                              recurrent_regularizer=regularizers.l2(self.l2)))(x)
        model = Model(x_input_sen, output_doc)
        return model

