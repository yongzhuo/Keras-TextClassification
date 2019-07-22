# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of CapsuleNet


from keras_textclassification.keras_layers.capsule import Capsule_bojone, CapsuleLayer, PrimaryCap, Length, Mask
from keras.layers import Conv1D, Conv2D, MaxPool2D, Concatenate, SpatialDropout1D
from keras.layers import Bidirectional, LSTM, GRU
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.layers import Reshape
from keras.models import Model
from keras import backend as K
import numpy as np

from keras_textclassification.base.graph import graph


class CapsuleNetGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.routings = hyper_parameters['model'].get('routings', 5)
        self.dim_capsule = hyper_parameters['model'].get('dim_capsule', 16)
        self.num_capsule = hyper_parameters['model'].get('num_capsule', 16)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output
        embed_layer = SpatialDropout1D(self.dropout)(embedding)
        conv_pools = []
        for filter in self.filters:
            x = Conv1D(filters=self.filters_num,
                       kernel_size=filter,
                       padding='valid',
                       kernel_initializer='normal',
                       activation='relu',
                       )(embed_layer)
            capsule = Capsule_bojone(num_capsule=self.num_capsule,
                                  dim_capsule=self.dim_capsule,
                                  routings=self.routings,
                                  kernel_size=(filter, 1),
                                  share_weights=True)(x)
            conv_pools.append(capsule)
        capsule = Concatenate(axis=-1)(conv_pools)
        capsule = Flatten()(capsule)
        capsule = Dropout(self.dropout)(capsule)
        output = Dense(self.label, activation=self.activate_classify)(capsule)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def create_model_gru(self, hyper_parameters):
        """
            构建神经网络, bi-gru + capsule
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output
        embed_layer = SpatialDropout1D(self.dropout)(embedding)
        x_bi = Bidirectional(GRU(self.filters_num,
                              activation='relu',
                              dropout=self.dropout,
                              recurrent_dropout=self.dropout,
                              return_sequences=True))(embed_layer)
        # 一层
        capsule = Capsule_bojone(num_capsule=self.num_capsule,
                              dim_capsule=self.dim_capsule,
                              routings=self.routings,
                              kernel_size=(3, 1),
                              share_weights=True)(x_bi)

        # # pooling多层
        # conv_pools = []
        # for filter in self.filters:
        #     capsule = Capsule_bojone(num_capsule=self.num_capsule,
        #                              dim_capsule=self.dim_capsule,
        #                              routings=self.routings,
        #                              kernel_size=(filter, 1),
        #                              share_weights=True)(x_bi)
        #     conv_pools.append(capsule)
        # capsule = Concatenate(axis=-1)(conv_pools)

        capsule = Flatten()(capsule)
        capsule = Dropout(self.dropout)(capsule)
        output = Dense(self.label, activation=self.activate_classify)(capsule)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)

    def create_model_basic(self, hyper_parameters):
        """
            构建神经网络, 原版, 速度超级慢
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output
        embedding = SpatialDropout1D(self.dropout)(embedding)
        embedding_reshape = Reshape((self.len_max, self.embed_size, 1))(embedding)
        conv_pools = []
        for filter in self.filters:
            # Layer 1: Just a conventional Conv2D layer
            conv1 = Conv2D(filters=self.filters_num,
                           kernel_size=(filter, self.embed_size),
                           strides=(1, 1),
                           padding='valid',
                           activation='relu',)(embedding_reshape)
            conv_shape = K.int_shape(conv1)
            conv1 = Reshape((conv_shape[1], conv_shape[3], conv_shape[2]))(conv1)
            # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
            primarycaps = PrimaryCap(inputs=conv1,
                                     dim_capsule=self.dim_capsule,
                                     n_channels=self.channel_size,
                                     kernel_size=(filter, conv_shape[3]),
                                     strides=(1, 1),
                                     padding='valid')
            # Layer 3: Capsule layer. Routing algorithm works here.
            digitcaps = CapsuleLayer(num_capsule=self.num_capsule, #self.label,
                                     dim_capsule=int(self.dim_capsule * 2),
                                     routings=self.routings, )(primarycaps)
            conv_pools.append(digitcaps)
        # 拼接
        x = Concatenate(axis=-1)(conv_pools)
        x = Flatten()(x)
        x = Dropout(self.dropout)(x)
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        # out_caps = [dense_layer]
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        # out_caps = Length(name='capsnet')(x)


        # 最后就是softmax
        self.model = Model(inputs=self.word_embedding.input, outputs=dense_layer)
        self.model.summary(120)

    def create_model_margin(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output
        embedding_reshape = Reshape((self.len_max, self.embed_size, 1))(embedding)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = Conv2D(filters=256,
                       kernel_size=3,
                       strides=1,
                       padding='valid',
                       activation='relu',)(embedding_reshape)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(inputs=conv1,
                                 dim_capsule=2,
                                 n_channels=6,
                                 kernel_size=2,
                                 strides=2,
                                 padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        x = CapsuleLayer(num_capsule=self.label,
                         dim_capsule=16,
                         routings=self.routings,)(primarycaps)
        # 拼接
        x = Dropout(self.dropout)(x)
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='capsnet')(x)

        self.model = Model(inputs=self.word_embedding.input, outputs=out_caps)
        self.model.summary(120)

    def margin_loss(self, y_true, y_pred):
        """
        Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
        """
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    # def create_compile(self):
    #     """
    #       构建优化器、损失函数和评价函数
    #     :return:
    #     """
    #     self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
    #                           loss=[self.margin_loss],
    #                           # loss_weights=[1., 0.392], # "The coefficient for the loss of decoder"分类不需要重构看结果
    #                           metrics=['accuracy'])
