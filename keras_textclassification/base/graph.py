# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of base


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
import numpy as np

# keras, tensorflow控制GPU使用率等
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.46
sess = tf.Session(config=config)
K.set_session(sess)

class graph:
    def __init__(self, hyper_parameters):
        """
            模型初始化
        :param hyper_parameters:json, json['model'] and json['embedding']  
        """
        hyper_parameters_model = hyper_parameters['model']
        self.label = hyper_parameters_model.get('label', 2)  # 类型
        self.batch_size = hyper_parameters_model.get('batch_size', 32)  # 批向量
        self.embed_size = hyper_parameters_model.get('embed_size', 300)  # 嵌入层尺寸
        self.filters = hyper_parameters_model.get('filters', [3, 4, 5])  # 卷积核大小
        self.filters_num = hyper_parameters_model.get('filters_num', 300)  # 核长
        self.channel_size = hyper_parameters_model.get('channel_size', 1)  # 通道数
        self.dropout = hyper_parameters_model.get('dropout', 0.5)          # dropout层系数，舍弃
        self.decay_step = hyper_parameters_model.get('decay_step', 100)    # 衰减步数
        self.decay_rate = hyper_parameters_model.get('decay_rate', 0.9)    # 衰减系数
        self.epochs = hyper_parameters_model.get('epochs', 20)             # 训练轮次
        self.len_max = hyper_parameters_model.get('len_max', 50)           # 文本最大长度
        self.vocab_size = hyper_parameters_model.get('vocab_size', 20000)  # 字典词典大小
        self.lr = hyper_parameters_model.get('lr', 1e-3)                   # 学习率
        self.l2 = hyper_parameters_model.get('l2', 1e-6)                   # l2正则化系数
        self.activate_classify = hyper_parameters_model.get('activate_classify', 'softmax')  # 分类激活函数,softmax或者signod
        self.embedding_type = hyper_parameters_model.get('embedding_type', 'word2vec')  #词嵌入方式，可以选择'bert'、'gpt-2'、'word2vec'或者'None'
        self.is_training = hyper_parameters_model.get('is_training', False)  # 是否训练
        self.model_path = hyper_parameters_model.get('model_path', "model")  # 模型地址
        self.create_model(hyper_parameters)
        if self.is_training:
            self.create_compile()


    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters: json，超参数
        :return:  
        """
        # embeddings选择
        Embeddings = None
        if self.embedding_type == 'random':
            from keras_textclassification.base.embedding import RandomEmbedding as Embeddings
        elif self.embedding_type == 'char':
            from keras_textclassification.base.embedding import CharEmbedding3 as Embeddings
        elif self.embedding_type == 'bert':
            from keras_textclassification.base.embedding import BertEmbedding as Embeddings
        elif self.embedding_type == 'word2vec':
            from keras_textclassification.base.embedding import WordEmbedding as Embeddings
        else:
            raise RuntimeError("your input embedding_type is wrong, it must be 'random'、 'bert' or 'word2vec")
        # 构建网络层
        self.word_embedding = Embeddings(hyper_parameters=hyper_parameters)
        self.model = None

    def callback(self):
        """
          评价函数、早停
        :return: 
        """
        cb_em = [ EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=3),
                  ModelCheckpoint(monitor='val_loss', mode='min', filepath=self.model_path, verbose=1,
                                  save_best_only=True, save_weights_only=False),]
        return cb_em

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """
        self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, x_dev, y_dev):
        """
            训练
        :param x_train: 
        :param y_train: 
        :param x_dev: 
        :param y_dev: 
        :return: 
        """
        self.model.fit(x_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=self.callback())

    def load_model(self):
        """
          模型下载
        :return: 
        """
        print("load_model start!")
        self.model.load_weights(self.model_path)
        print("load_model end!")

    def predict(self, sen):
        """
          预测
        :param sen: 
        :return: 
        """
        if type(sen)==np.ndarray:
            sen = sen
        elif type(sen)==list:
            sen = np.array([sen])
        else:
            raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
        return self.model.predict(sen)

