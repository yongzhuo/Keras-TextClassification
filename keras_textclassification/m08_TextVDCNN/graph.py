# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 11:45
# @author   :Mo
# @function :graph of dcnn
# @paper:    Very Deep Convolutional Networks(https://www.aclweb.org/anthology/E17-1104)


from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Dense, Lambda
from keras.layers import Dropout, Reshape, Concatenate
from keras.layers import Layer
from keras.layers import Flatten
from keras.layers import LeakyReLU, PReLU, ReLU
from keras.layers import Add, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras_textclassification.base.graph import graph
import keras.backend as K

import tensorflow as tf


class VDCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.l2 = hyper_parameters['model'].get('l2', 0.0000032)
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        self.activation_conv = hyper_parameters['model'].get('activation_conv', 'linear')
        self.pool_type = hyper_parameters['model'].get('pool_type', 'max')
        self.shortcut = hyper_parameters['model'].get('shortcut', True)
        self.top_k = hyper_parameters['model'].get('top_k', 2)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        embedding_output_spatial = SpatialDropout1D(self.dropout_spatial)(embedding_output)

        # 首先是 region embedding 层
        conv_1 = Conv1D(self.filters[0][0],
                        kernel_size=1,
                        strides=1,
                        padding='SAME',
                        kernel_regularizer=l2(self.l2),
                        bias_regularizer=l2(self.l2),
                        activation=self.activation_conv,
                        )(embedding_output_spatial)
        block = ReLU()(conv_1)

        for filters_block in self.filters:
            for j in range(filters_block[1]-1):
                # conv + short-cut
                block_mid = self.convolutional_block(block, units=filters_block[0])
                block = shortcut_conv(block, block_mid, shortcut=True)
            # 这里是conv + max-pooling
            block_mid = self.convolutional_block(block, units=filters_block[0])
            block = shortcut_pool(block, block_mid, filters=filters_block[0], pool_type=self.pool_type, shortcut=True)

        block = k_max_pooling(top_k=self.top_k)(block)
        block = Flatten()(block)
        block = Dropout(self.dropout)(block)
        # 全连接层
        # block_fully = Dense(2048, activation='tanh')(block)
        # output = Dense(2048, activation='tanh')(block_fully)
        output = Dense(self.label, activation=self.activate_classify)(block)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def convolutional_block(self, inputs, units=256):
        """
            Each convolutional block (see Figure 2) is a sequence of two convolutional layers, 
            each one followed by a temporal BatchNorm (Ioffe and Szegedy, 2015) layer and an ReLU activation. 
            The kernel size of all the temporal convolutions is 3, 
            with padding such that the temporal resolution is preserved 
            (or halved in the case of the convolutional pooling with stride 2, see below). 
        :param inputs: tensor, input
        :param units: int, units
        :return: tensor, result of convolutional block
        """
        x = Conv1D(units,
                    kernel_size=3,
                    padding='SAME',
                    strides=1,
                    kernel_regularizer=l2(self.l2),
                    bias_regularizer=l2(self.l2),
                    activation=self.activation_conv,
                    )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(units,
                    kernel_size=3,
                    strides=1,
                    padding='SAME',
                    kernel_regularizer=l2(self.l2),
                    bias_regularizer=l2(self.l2),
                    activation=self.activation_conv,
                    )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x


def shortcut_pool(inputs, output, filters=256, pool_type='max', shortcut=True):
    """
        ResNet(shortcut连接|skip连接|residual连接), 
        这里是用shortcut连接. 恒等映射, block+f(block)
        再加上 downsampling实现
        参考: https://github.com/zonetrooper32/VDCNN/blob/keras_version/vdcnn.py
    :param inputs: tensor
    :param output: tensor
    :param filters: int
    :param pool_type: str, 'max'、'k-max' or 'conv' or other
    :param shortcut: boolean
    :return: tensor
    """
    if shortcut:
        conv_2 = Conv1D(filters=filters, kernel_size=1, strides=2, padding='SAME')(inputs)
        conv_2 = BatchNormalization()(conv_2)
        output = downsampling(output, pool_type=pool_type)
        out = Add()([output, conv_2])
    else:
        out = ReLU(inputs)
        out = downsampling(out, pool_type=pool_type)
    if pool_type is not None: # filters翻倍
        out = Conv1D(filters=filters*2, kernel_size=1, strides=1, padding='SAME')(out)
        out = BatchNormalization()(out)
    return out

def shortcut_conv(inputs, output, shortcut=True):
    """
        shortcut of conv
    :param inputs: tensor
    :param output: tensor
    :param shortcut: boolean
    :return: tensor
    """
    if shortcut:
        output = Add()([output, inputs])
    return output

def downsampling(inputs, pool_type='max'):
    """
        In addition, downsampling with stride 2 essentially doubles the effective coverage 
        (i.e., coverage in the original document) of the convolution kernel; 
        therefore, after going through downsampling L times, 
        associations among words within a distance in the order of 2L can be represented. 
        Thus, deep pyramid CNN is computationally efﬁcient for representing long-range associations 
        and so more global information. 
        参考: https://github.com/zonetrooper32/VDCNN/blob/keras_version/vdcnn.py
    :param inputs: tensor,
    :param pool_type: str, select 'max', 'k-max' or 'conv'
    :return: tensor,
    """
    if pool_type == 'max':
        output = MaxPooling1D(pool_size=3, strides=2, padding='SAME')(inputs)
    elif pool_type == 'k-max':
        output = k_max_pooling(top_k=int(K.int_shape(inputs)[1]/2))(inputs)
    elif pool_type == 'conv':
        output = Conv1D(kernel_size=3, strides=2, padding='SAME')(inputs)
    else:
        output = MaxPooling1D(pool_size=3, strides=2, padding='SAME')(inputs)
    return output

class k_max_pooling(Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=8, **kwargs):
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

