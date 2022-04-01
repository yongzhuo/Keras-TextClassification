# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/2/20 15:52
# @author  : Mo
# @function:


from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
from keras import backend as K

from keras.layers import Reshape, Concatenate, Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from keras_textclassification.base.graph import graph


class TextCNNGraphTFT(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        """
                    模型初始化
                :param hyper_parameters:json, json['model'] and json['embedding']  
                """
        self.len_max = hyper_parameters.get('len_max', 50)  # 文本最大长度
        self.embed_size = hyper_parameters.get('embed_size', 300)  # 嵌入层尺寸
        self.trainable = hyper_parameters.get('trainable', False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')  # 词嵌入方式，可以选择'xlnet'、'bert'、'gpt-2'、'word2vec'或者'None'
        self.gpu_memory_fraction = hyper_parameters.get('gpu_memory_fraction', None)  # gpu使用率, 默认不配置
        self.hyper_parameters = hyper_parameters
        hyper_parameters_model = hyper_parameters['model']
        self.label = hyper_parameters_model.get('label', 2)  # 类型
        self.batch_size = hyper_parameters_model.get('batch_size', 32)  # 批向量
        self.filters = hyper_parameters_model.get('filters', [3, 4, 5])  # 卷积核大小
        self.filters_num = hyper_parameters_model.get('filters_num', 300)  # 核数
        self.channel_size = hyper_parameters_model.get('channel_size', 1)  # 通道数
        self.dropout = hyper_parameters_model.get('dropout', 0.5)  # dropout层系数，舍弃
        self.decay_step = hyper_parameters_model.get('decay_step', 100)  # 衰减步数
        self.decay_rate = hyper_parameters_model.get('decay_rate', 0.9)  # 衰减系数
        self.epochs = hyper_parameters_model.get('epochs', 20)  # 训练轮次
        self.vocab_size = hyper_parameters_model.get('vocab_size', 20000)  # 字典词典大小
        self.lr = hyper_parameters_model.get('lr', 1e-3)  # 学习率
        self.l2 = hyper_parameters_model.get('l2', 1e-6)  # l2正则化系数
        self.activate_classify = hyper_parameters_model.get('activate_classify', 'softmax')  # 分类激活函数,softmax或者signod
        self.loss = hyper_parameters_model.get('loss',
                                               'categorical_crossentropy')  # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
        self.metrics = hyper_parameters_model.get('metrics',
                                                  'accuracy')  # acc, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
        self.is_training = hyper_parameters_model.get('is_training', False)  # 是否训练, 保存时候为Flase,方便预测
        self.path_model_dir = hyper_parameters_model.get('path_model_dir', path_model_dir)  # 模型目录地址
        self.model_path = hyper_parameters_model.get('model_path', path_model)  # 模型地址
        self.path_hyper_parameters = hyper_parameters_model.get('path_hyper_parameters',
                                                                path_hyper_parameters)  # 超参数保存地址
        self.path_fineture = hyper_parameters_model.get('path_fineture',
                                                        path_fineture)  # embedding层保存地址, 例如静态词向量、动态词向量、微调bert层等
        self.patience = hyper_parameters_model.get('patience', 3)  # 早停, 2-3就可以了
        self.optimizer_name = hyper_parameters_model.get('optimizer_name', 'Adam')  # 早停, 2-3就可以了
        if self.gpu_memory_fraction:
            # keras, tensorflow控制GPU使用率等
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
            # config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)

    def create_model(self, hyper_parameters, input_x):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """

        embedding = input_x
        embedding_reshape = Reshape((self.len_max, self.embed_size, 1))(embedding)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for filter in self.filters:
            conv = Conv2D(filters = self.filters_num,
                          kernel_size = (filter, self.embed_size),
                          padding = 'valid',
                          kernel_initializer = 'normal',
                          activation = 'relu',
                          )(embedding_reshape)
            pooled = MaxPool2D(pool_size = (self.len_max - filter + 1, 1),
                               strides = (1, 1),
                               padding = 'valid',
                               )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = Concatenate(axis=-1)(conv_pools)
        x = Flatten()(x)
        x = Dropout(self.dropout)(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)
