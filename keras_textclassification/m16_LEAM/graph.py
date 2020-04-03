# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2020/4/1 19:51
# @author   :Mo
# @function :graph of LEAM
# @paper    :Joint Embedding of Words and Labels for Text Classification(https://arxiv.org/abs/1805.04174)


from keras_textclassification.base.graph import graph
from keras.layers import Dense, Dropout, Concatenate
from keras.models import Model

from keras_textclassification.keras_layers.attention_dot import AttentionDot, CVG_Layer
from keras_textclassification.keras_layers.attention_self import AttentionSelf


class LEAMGraph(graph):
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
        # 构建网络层 sentence
        # self.word_embedding_attention = AttentionSelf(self.embed_size)(self.word_embedding.output)
        self.word_embedding_attention = AttentionDot()(self.word_embedding.output)

        # 1.C*V/G;  2.relu(text-cnn);  3.softmax;  4.β * V
        pools = []
        for filter in self.filters:
            x_cvg = CVG_Layer(self.embed_size, filter, self.label)(self.word_embedding_attention)
            pools.append(x_cvg)
        pools_concat = Concatenate()(pools)
        # MLP
        x_cvg_dense = Dense(int(self.embed_size/2), activation="relu")(pools_concat)
        x = Dropout(self.dropout)(x_cvg_dense)
        output = Dense(self.label, activation=self.activate_classify)(x)

        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(132)
