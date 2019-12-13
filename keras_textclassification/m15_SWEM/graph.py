# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/12/13 10:51
# @author   :Mo
# @function :graph of SWEM
# @paper    : Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms()


from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Concatenate
from keras_textclassification.base.graph import graph
from keras.layers import Dense, Lambda
import tensorflow as tf
from keras.models import Model
import keras.backend as K


class SWEMGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.encode_type = hyper_parameters["model"].get("encode_type", "MAX") # AVG, CONCAT, HIERARCHICAL
        self.n_win = hyper_parameters["model"].get("n_win", 3) # n_win=3
        super().__init__(hyper_parameters)


    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output

        def win_mean(x):
            res_list = []
            for i in range(self.len_max-self.n_win+1):
                x_mean = tf.reduce_mean(x[:, i:i + self.n_win, :], axis=1)
                x_mean_dims = tf.expand_dims(x_mean, axis=-1)
                res_list.append(x_mean_dims)
            res_list = tf.concat(res_list, axis=-1)
            gg = tf.reduce_max(res_list, axis=-1)
            return gg

        if self.encode_type=="HIERARCHICAL":
            x = Lambda(win_mean, output_shape=(self.embed_size, ))(embedding)
        elif self.encode_type=="MAX":
            x = GlobalMaxPooling1D()(embedding)
        elif self.encode_type=="AVG":
            x = GlobalAveragePooling1D()(embedding)
        elif self.encode_type == "CONCAT":
            x_max = GlobalMaxPooling1D()(embedding)
            x_avg = GlobalAveragePooling1D()(embedding)
            x = Concatenate()([x_max, x_avg])
        else:
            raise RuntimeError("encode_type must be 'MAX', 'AVG', 'CONCAT', 'HIERARCHICAL'")

        output = Dense(self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(132)

