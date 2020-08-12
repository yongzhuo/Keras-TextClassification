# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of base


from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
from keras_textclassification.data_preprocess.generator_preprocess import PreprocessGenerator, PreprocessSimGenerator
from keras_textclassification.data_preprocess.text_preprocess import save_json
from keras_textclassification.keras_layers.keras_lookahead import Lookahead
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_textclassification.keras_layers.keras_radam import RAdam
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
import os


class graph:
    def __init__(self, hyper_parameters):
        """
            模型初始化
        :param hyper_parameters:json, json['model'] and json['embedding']  
        """
        self.len_max = hyper_parameters.get('len_max', 50)           # 文本最大长度
        self.embed_size = hyper_parameters.get('embed_size', 300)  # 嵌入层尺寸
        self.trainable = hyper_parameters.get('trainable', False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')  # 词嵌入方式，可以选择'xlnet'、'bert'、'gpt-2'、'word2vec'或者'None'
        self.gpu_memory_fraction = hyper_parameters.get('gpu_memory_fraction', None) # gpu使用率, 默认不配置
        self.hyper_parameters = hyper_parameters
        hyper_parameters_model = hyper_parameters['model']
        self.label = hyper_parameters_model.get('label', 2)  # 类型
        self.batch_size = hyper_parameters_model.get('batch_size', 32)  # 批向量
        self.filters = hyper_parameters_model.get('filters', [3, 4, 5])  # 卷积核大小
        self.filters_num = hyper_parameters_model.get('filters_num', 300)  # 核数
        self.channel_size = hyper_parameters_model.get('channel_size', 1)  # 通道数
        self.dropout = hyper_parameters_model.get('dropout', 0.5)          # dropout层系数，舍弃
        self.decay_step = hyper_parameters_model.get('decay_step', 100)    # 衰减步数
        self.decay_rate = hyper_parameters_model.get('decay_rate', 0.9)    # 衰减系数
        self.epochs = hyper_parameters_model.get('epochs', 20)             # 训练轮次
        self.vocab_size = hyper_parameters_model.get('vocab_size', 20000)  # 字典词典大小
        self.lr = hyper_parameters_model.get('lr', 1e-3)                   # 学习率
        self.l2 = hyper_parameters_model.get('l2', 1e-6)                   # l2正则化系数
        self.activate_classify = hyper_parameters_model.get('activate_classify', 'softmax')  # 分类激活函数,softmax或者signod
        self.loss = hyper_parameters_model.get('loss', 'categorical_crossentropy') # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
        self.metrics = hyper_parameters_model.get('metrics', 'accuracy') # acc, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
        self.is_training = hyper_parameters_model.get('is_training', False)  # 是否训练, 保存时候为Flase,方便预测
        self.path_model_dir = hyper_parameters_model.get('path_model_dir', path_model_dir)  # 模型目录地址
        self.model_path = hyper_parameters_model.get('model_path', path_model)  # 模型地址
        self.path_hyper_parameters = hyper_parameters_model.get('path_hyper_parameters', path_hyper_parameters) # 超参数保存地址
        self.path_fineture = hyper_parameters_model.get('path_fineture', path_fineture) # embedding层保存地址, 例如静态词向量、动态词向量、微调bert层等
        self.patience = hyper_parameters_model.get('patience', 3) # 早停, 2-3就可以了
        self.optimizer_name = hyper_parameters_model.get('optimizer_name', 'Adam') # 早停, 2-3就可以了
        if self.gpu_memory_fraction:
            # keras, tensorflow控制GPU使用率等
            import tensorflow as tf
            config = tf.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
        self.create_model(hyper_parameters)
        if self.is_training: # 是否是训练阶段, 与预测区分开
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
        elif self.embedding_type == 'bert':
            from keras_textclassification.base.embedding import BertEmbedding as Embeddings
        elif self.embedding_type == 'xlnet':
            from keras_textclassification.base.embedding import XlnetEmbedding as Embeddings
        elif self.embedding_type == 'albert':
            from keras_textclassification.base.embedding import AlbertEmbedding as Embeddings
        elif self.embedding_type == 'word2vec':
            from keras_textclassification.base.embedding import WordEmbedding as Embeddings
        else:
            raise RuntimeError("your input embedding_type is wrong, it must be 'xlnet'、'random'、 'bert'、 'albert' or 'word2vec")
        # 构建网络层
        self.word_embedding = Embeddings(hyper_parameters=hyper_parameters)
        if os.path.exists(self.path_fineture) and self.trainable:
            self.word_embedding.model.load_weights(self.path_fineture)
            print("load path_fineture ok!")
        self.model = None

    def callback(self):
        """
          评价函数、早停
        :return: 
        """
        cb_em = [ TensorBoard(log_dir=os.path.join(self.path_model_dir, "logs"), batch_size=self.batch_size, update_freq='batch'),
                  EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=self.patience),
                  ModelCheckpoint(monitor='val_loss', mode='min', filepath=self.model_path, verbose=1,
                                  save_best_only=True, save_weights_only=True),]
        return cb_em

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """

        if self.optimizer_name.upper() == "ADAM":
            self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss= self.loss,
                               metrics=[self.metrics]) # Any optimize
        elif self.optimizer_name.upper() == "RADAM":
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss,
                               metrics=[self.metrics]) # Any optimize
        else:
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss= self.loss,
                               metrics=[self.metrics]) # Any optimize
            lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            lookahead.inject(self.model)  # add into model

    def fit(self, x_train, y_train, x_dev, y_dev):
        """
            训练
        :param x_train: 
        :param y_train: 
        :param x_dev: 
        :param y_dev: 
        :return: 
        """
        # 保存超参数
        self.hyper_parameters['model']['is_training'] = False # 预测时候这些设为False
        self.hyper_parameters['model']['trainable'] = False
        self.hyper_parameters['model']['dropout'] = 0.0

        save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)
        # if self.is_training and os.path.exists(self.model_path):
        #     print("load_weights")
        #     self.model.load_weights(self.model_path)
        # 训练模型
        self.model.fit(x_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=self.callback())
        # 保存embedding, 动态的
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def fit_generator(self, embed, rate=1):
        """

        :param data_fit_generator: yield, 训练数据
        :param data_dev_generator: yield, 验证数据
        :param steps_per_epoch: int, 训练一轮步数
        :param validation_steps: int, 验证一轮步数
        :return: 
        """
        # 保存超参数
        self.hyper_parameters['model']['is_training'] = False  # 预测时候这些设为False
        self.hyper_parameters['model']['trainable'] = False
        self.hyper_parameters['model']['dropout'] = 0.0

        save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)

        pg = PreprocessGenerator(self.path_model_dir)
        _, len_train = pg.preprocess_get_label_set(self.hyper_parameters['data']['train_data'])
        data_fit_generator = pg.preprocess_label_ques_to_idx(embedding_type=self.hyper_parameters['embedding_type'],
                                                             batch_size=self.batch_size,
                                                             path=self.hyper_parameters['data']['train_data'],
                                                             epcoh=self.epochs,
                                                             embed=embed,
                                                             rate=rate)
        _, len_val = pg.preprocess_get_label_set(self.hyper_parameters['data']['val_data'])
        data_dev_generator = pg.preprocess_label_ques_to_idx(embedding_type=self.hyper_parameters['embedding_type'],
                                                             batch_size=self.batch_size,
                                                             path=self.hyper_parameters['data']['val_data'],
                                                             epcoh=self.epochs,
                                                             embed=embed,
                                                             rate=rate)
        steps_per_epoch = len_train // self.batch_size + 1
        validation_steps = len_val // self.batch_size + 1
        # 训练模型
        self.model.fit_generator(generator=data_fit_generator,
                                 validation_data=data_dev_generator,
                                 callbacks=self.callback(),
                                 epochs=self.epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)
        # 保存embedding, 动态的
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def fit_generator_sim(self, embed, rate=1):
        """

        :param data_fit_generator: yield, 训练数据
        :param data_dev_generator: yield, 验证数据
        :param steps_per_epoch: int, 训练一轮步数
        :param validation_steps: int, 验证一轮步数
        :return: 
        """
        # 保存超参数
        self.hyper_parameters['model']['is_training'] = False  # 预测时候这些设为False
        self.hyper_parameters['model']['trainable'] = False
        self.hyper_parameters['model']['dropout'] = 0.0

        save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)

        pg = PreprocessSimGenerator(self.hyper_parameters['model']['path_model_dir'])
        _, len_train = pg.preprocess_get_label_set(self.hyper_parameters['data']['train_data'])
        data_fit_generator = pg.preprocess_label_ques_to_idx(embedding_type=self.hyper_parameters['embedding_type'],
                                                             batch_size=self.batch_size,
                                                             path=self.hyper_parameters['data']['train_data'],
                                                             embed=embed,
                                                             epcoh=self.epochs,
                                                             rate=rate)
        _, len_val = pg.preprocess_get_label_set(self.hyper_parameters['data']['val_data'])
        data_dev_generator = pg.preprocess_label_ques_to_idx(embedding_type=self.hyper_parameters['embedding_type'],
                                                             batch_size=self.batch_size,
                                                             path=self.hyper_parameters['data']['val_data'],
                                                             embed=embed,
                                                             epcoh=self.epochs,
                                                             rate=rate)
        steps_per_epoch = len_train // self.batch_size + 1
        validation_steps = len_val // self.batch_size + 1
        # self.model.load_weights(self.model_path)
        # 训练模型
        self.model.fit_generator(generator=data_fit_generator,
                                 validation_data=data_dev_generator,
                                 callbacks=self.callback(),
                                 epochs=self.epochs,
                                 steps_per_epoch=32,
                                 validation_steps=6)
        # 保存embedding, 动态的
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)
    # 1600000/6=266666
    # 300000/6=50000

    # 36000/6000
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
        if self.embedding_type in ['bert', 'xlnet', 'albert']:
            if type(sen) == np.ndarray:
                sen = sen.tolist()
            elif type(sen) == list:
                sen = sen
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sen)
        else:
            if type(sen)==np.ndarray:
                sen = sen
            elif type(sen)==list:
                sen = np.array([sen])
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sen)


