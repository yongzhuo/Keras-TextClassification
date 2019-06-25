# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :train of textcnn-char-random with baidu-qa-2019 in question title

import time

from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph

if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 256,
                                     'embed_size': 300,
                                     'filters': [2, 3, 4],
                                     'kernel_size': 600,
                                     'channel_size': 1,
                                     'dropout': 0.5,
                                     'decay_step': 100,
                                     'decay_rate': 0.9,
                                     'epochs': 20,
                                     'len_max': 50,
                                     'vocab_size': 20000, #这里随便填的，会根据代码里修改
                                     'lr': 1e-3,
                                     'l2': 1e-6,
                                     'activate_classify': 'softmax',
                                     'embedding_type': 'random', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,},
                        'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,# char可以自己设置, 文件名必须带"term"字符
                                      'level_type': 'char',
                                      'embed_size': 300,
                                      'len_max': 50,},
                         }

    time_start  = time.time()

    graph = Graph(hyper_parameters)
    ra_ed = graph.word_embedding
    pt = PreprocessText()
    # rate=0 - 1
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed, rate=1)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed, rate=1)
    print(len(y_train))
    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time()-time_start))

# 1425170/1425170 [==============================] - 648s 455us/step - loss: 0.8771 - acc: 0.7317 - val_loss: 1.2533 - val_acc: 0.7069
# Epoch 00001: val_loss improved from inf to 1.25335, saving model to D:\workspace\pythonMyCode\django_project\ClassificationTextChinese/data/model/fast_text/model_fast_text.f5
# Epoch 2/20
# TIME: 20 * 5轮= 100 (min)
# acc: 0.7317
