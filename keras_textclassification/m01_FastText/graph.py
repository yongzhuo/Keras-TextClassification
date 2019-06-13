# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of fasttext
# @paper: Bag of Tricks for Efﬁcient Text Classiﬁcation(https://arxiv.org/abs/1607.01759)


from keras_textclassification.base.graph import graph
from keras.layers import Dense
from keras.layers import GlobalMaxPooling1D
from keras.models import Model


class FastTextGraph(graph):
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
        x = GlobalMaxPooling1D()(embedding)
        output = Dense(self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)


