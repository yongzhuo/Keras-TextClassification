# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 11:45
# @author   :Mo
# @function :graph of dcnn
# @paper:    Deep Pyramid Convolutional Neural Networks for Text Categorization(https://www.aclweb.org/anthology/P17-1052)


from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Add, Dense, Dropout, Flatten
from keras.layers import LeakyReLU, PReLU, ReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
from keras_textclassification.base.graph import graph



class DPCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.l2 = hyper_parameters['model'].get('l2', 0.0000032)
        self.pooling_size_strides = hyper_parameters['model'].get('pooling_size_strides', [3, 2])
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        self.activation_conv = hyper_parameters['model'].get('activation_conv', 'linear')
        self.layer_repeats = hyper_parameters['model'].get('layer_repeats', 5)
        self.full_connect_unit = hyper_parameters['model'].get('self.full_connect_unit', 256)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络, 参考 https://blog.csdn.net/dqcfkyqdxym3f8rb0/article/details/86662906
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding_output = self.word_embedding.output
        embedding_output_spatial = SpatialDropout1D(self.dropout_spatial)(embedding_output)

        # 首先是 region embedding 层
        conv_1 = Conv1D(self.filters_num,
                            kernel_size=1,
                            padding='SAME',
                            kernel_regularizer=l2(self.l2),
                            bias_regularizer=l2(self.l2),
                            activation=self.activation_conv,
                            )(embedding_output_spatial)
        conv_1_prelu = PReLU()(conv_1)
        block = None
        layer_curr = 0
        for i in range(self.layer_repeats):
            if i == 0: # 第一层输入用embedding输出的结果作为输入
                block = self.ResCNN(embedding_output_spatial)
                block_add = Add()([block, conv_1_prelu])
                block = MaxPooling1D(pool_size=self.pooling_size_strides[0],
                                     strides=self.pooling_size_strides[1])(block_add)
            elif self.layer_repeats - 1 == i or layer_curr == 1: # 最后一次repeat用GlobalMaxPooling1D
                block_last = self.ResCNN(block)
                # ResNet(shortcut连接|skip连接|residual连接), 这里是shortcut连接. 恒等映射, block+f(block)
                block_add = Add()([block_last, block])
                block = GlobalMaxPooling1D()(block_add)
                break
            else: # 中间层 repeat
                if K.int_shape(block)[1] // 2 < 8: # 防止错误, 不能pooling/2的情况, 就是说size >= 2
                    layer_curr = 1
                block_mid = self.ResCNN(block)
                block_add = Add()([block_mid, block])
                block = MaxPooling1D(pool_size=self.pooling_size_strides[0],
                                     strides=self.pooling_size_strides[1])(block_add)

        # 全连接层
        output = Dense(self.full_connect_unit, activation='linear')(block)
        output = BatchNormalization()(output)
        #output = PReLU()(output)
        output = Dropout(self.dropout)(output)
        output = Dense(self.label, activation=self.activate_classify)(output)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def ResCNN(self, x):
        """
            repeat of two conv
        :param x: tensor, input shape
        :return: tensor, result of two conv of resnet
        """
        # pre-activation
        # x = PReLU()(x)
        x = Conv1D(self.filters_num,
                                kernel_size=1,
                                padding='SAME',
                                kernel_regularizer=l2(self.l2),
                                bias_regularizer=l2(self.l2),
                                activation=self.activation_conv,
                                )(x)
        x = BatchNormalization()(x)
        #x = PReLU()(x)
        x = Conv1D(self.filters_num,
                                kernel_size=1,
                                padding='SAME',
                                kernel_regularizer=l2(self.l2),
                                bias_regularizer=l2(self.l2),
                                activation=self.activation_conv,
                                )(x)
        x = BatchNormalization()(x)
        # x = Dropout(self.dropout)(x)
        x = PReLU()(x)
        return x
