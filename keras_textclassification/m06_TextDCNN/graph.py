# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 11:45
# @author   :Mo
# @function :graph of dcnn
# @paper:    A Convolutional Neural Network for Modelling Sentences(http://www.aclweb.org/anthology/P14-1062)


from keras.layers import Conv1D, MaxPooling1D
from keras.layers import ZeroPadding1D
from keras.layers import Dense, Lambda
from keras.layers import Dropout, Reshape, Concatenate
from keras.layers import Layer
from keras.layers import Flatten
from keras.layers import Add
from keras.models import Model
from keras.optimizers import Adam

from keras_textclassification.base.graph import graph
import tensorflow as tf


class DCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        # self.top_ks = hyper_parameters['model'].get('top_ks', [[6, 3], [7, 4], [9, 3]])
        super().__init__(hyper_parameters)


    def create_model_2(self, hyper_parameters):
        """
            构建神经网络，只有2层静态
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        pools = []
        for i in range(len(self.filters)):

            # 第一个,宽卷积,动态k-max池化
            conv_1 = wide_convolution(name="wide_convolution_{}".format(i), filter_num=self.filters_num, filter_size=self.filters[i][0])(embedding_output)
            top_k_1 = select_k(self.len_max, len(self.filters[i]), 1) # 求取k
            dynamic_k_max_pooled_1 = dynamic_k_max_pooling(top_k=top_k_1)(conv_1)
            # 第二个,宽卷积,动态k-max池化
            conv_2 = wide_convolution(name="wide_convolution_{}_{}".format(i, i), filter_num=self.filters_num, filter_size=self.filters[i][1])(dynamic_k_max_pooled_1)
            fold_conv_2 = prem_fold()(conv_2)
            top_k_2 = select_k(self.len_max, len(self.filters[i]), 2)
            dynamic_k_max_pooled_2 = dynamic_k_max_pooling(top_k=top_k_2)(fold_conv_2)
            pools.append(dynamic_k_max_pooled_2)
        pools_concat = Concatenate(axis=1)(pools)
        pools_concat_dropout = Dropout(self.dropout)(pools_concat)
        x = Flatten()(pools_concat_dropout)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        # output = Dense(units=self.label, activation='linear')(pools_concat)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def create_model(self, hyper_parameters):
        """
            构建神经网络，只有3层静态
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        self.filters_num = self.word_embedding.embed_size
        pools = []
        for i in range(len(self.filters)):

            # 第一个,宽卷积,动态k-max池化
            conv_1 = wide_convolution(name="wide_convolution_{}".format(i),
                                      filter_num=self.filters_num, filter_size=self.filters[i][0])(embedding_output)
            top_k_1 = select_k(self.len_max, len(self.filters[i]), 1) # 求取k
            dynamic_k_max_pooled_1 = dynamic_k_max_pooling(top_k=top_k_1)(conv_1)
            # 第二个,宽卷积,动态k-max池化
            conv_2 = wide_convolution(name="wide_convolution_{}_{}".format(i, i),
                                      filter_num=self.filters_num, filter_size=self.filters[i][1])(dynamic_k_max_pooled_1)
            top_k_2 = select_k(self.len_max, len(self.filters[i]), 2)
            dynamic_k_max_pooled_2 = dynamic_k_max_pooling(top_k=top_k_2)(conv_2)
            # 第三层,宽卷积,Fold层,动态k-max池化
            conv_3 = wide_convolution(name="wide_convolution_{}_{}_{}".format(i, i, i), filter_num=self.filters_num,
                                      filter_size=self.filters[i][2])(dynamic_k_max_pooled_2)
            fold_conv_3 = prem_fold()(conv_3)
            top_k_3 = select_k(self.len_max, len(self.filters[i]), 3)  # 求取k
            dynamic_k_max_pooled_3 = dynamic_k_max_pooling(top_k=top_k_3)(fold_conv_3)
            pools.append(dynamic_k_max_pooled_3)
        pools_concat = Concatenate(axis=1)(pools)
        pools_concat_dropout = Dropout(self.dropout)(pools_concat)
        x = Flatten()(pools_concat_dropout)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        # output = Dense(units=self.label, activation='linear')(pools_concat)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)


class wide_convolution(Layer):
    """
        paper: http://www.aclweb.org/anthology/P14-1062
        paper title: "A Convolutional Neural Network for Modelling Sentences"
        宽卷积, 如果s表示句子最大长度, m为卷积核尺寸,
           则宽卷积输出为 s + m − 1,
           普通卷积输出为 s - m + 1.
        github keras实现可以参考: https://github.com/AlexYangLi/TextClassification/blob/master/models/keras_dcnn_model.py
    """
    def __init__(self, filter_num=300, filter_size=3, **kwargs):
        self.filter_size = filter_size
        self.filter_num = filter_num
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x_input_pad = ZeroPadding1D((self.filter_size-1, self.filter_size-1))(inputs)
        conv_1d = Conv1D(filters=self.filter_num,
                         kernel_size=self.filter_size,
                         strides=1,
                         padding='VALID',
                         kernel_initializer='normal', # )(x_input_pad)
                         activation='tanh')(x_input_pad)
        return conv_1d

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.filter_size - 1, input_shape[-1]


class dynamic_k_max_pooling(Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=3, **kwargs):
        self.top_k = top_k
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        inputs_reshape = tf.transpose(inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=inputs_reshape, k=self.top_k, sorted=False).values
        pool_top_k_reshape = tf.transpose(pool_top_k, perm=[0, 2, 1])
        return pool_top_k_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]


class prem_fold(Layer):
    """
        paper:       http://www.aclweb.org/anthology/P14-1062
        paper title: A Convolutional Neural Network for Modelling Sentences
        detail:      垂直于句子长度的方向，相邻值相加，就是embedding层300那里，（0+1,2+3...298+299）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, conv_shape):
        super().build(conv_shape)

    def call(self, convs):
        conv1 = convs[:, :, ::2]
        conv2 = convs[:, :, 1::2]
        conv_fold = Add()([conv1, conv2])
        return conv_fold

    def compute_output_shape(self, conv_shape):
        return conv_shape[0], conv_shape[1], int(conv_shape[2] / 2)


def select_k(len_max, length_conv, length_curr, k_con=3):
    """
        dynamic k max pooling中的k获取
    :param len_max:int, max length of input sentence 
    :param length_conv: int, deepth of all convolution layer
    :param length_curr: int, deepth of current convolution layer
    :param k_con: int, k of constant 
    :return: int, return 
    """
    if type(len_max) != int:
        len_max = len_max[0]
    if type(length_conv) != int:
        length_conv = length_conv[0]
    if length_conv >= length_curr:
        k_ml = int(len_max * (length_conv-length_curr) / length_conv)
        k = max(k_ml, k_con)
    else:
        k = k_con
    return k

