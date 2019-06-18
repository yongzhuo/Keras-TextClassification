# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of RCNNGraph_kim with baidu-qa-2019 in question title
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_embedding_bert
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m00_Bert.graph import BertGraph as Graph

import numpy as np
import random

if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 64,
                                     'embed_size': 30,
                                     'filters': [2, 3, 4],
                                     'filters_num': 30,
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
                                     'embedding_type': 'bert', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,
                                     'rnn_type': 'GRU', # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                     'rnn_units': 256,  # large 650, small is 300
                                     },
                        'embedding':{ 'embedding_type': 'bert',
                                      'corpus_path': path_embedding_bert,
                                      'level_type': 'char',
                                      'embed_size': 30,
                                      'len_max': 50,
                                      'layer_indexes': [12] # range 1 to 12
                                      },
                         }

    import time
    time_start = time.time()

    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    pt = PreprocessText()
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed, rate=0.01)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed, rate=0.01)
    indexs = [ids for ids in range(len(y_train))]
    random.shuffle(indexs)
    x_train, y_train = x_train[indexs], y_train[indexs]
    print("data propress ok 1 !")
    print(len(y_train))
    x_train = x_train.tolist()
    x_val = x_val.tolist()
    x_train_1 = np.array([x[0] for x in x_train])
    x_train_2 = np.array([x[1] for x in x_train])
    x_train = [x_train_1, x_train_2]
    x_val_1 = np.array([x[0] for x in x_val])
    x_val_2 = np.array([x[1] for x in x_val])
    x_val = [x_val_1, x_val_2]
    print("data propress ok 2 !")
    print(len(y_train))
    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time()-time_start))


# 1425170/1425170 [==============================] - 10032s 7ms/step - loss: 1.0163 - acc: 0.6989 - val_loss: 0.8111 - val_acc: 0.7703
# Epoch 00003: val_loss did not improve from 0.79959

# 1425170/1425170 [==============================] - 9219s 6ms/step - loss: 1.0105 - acc: 0.7015 - val_loss: 0.7999 - val_acc: 0.7741
# Epoch 00007: val_loss did not improve from 0.79582
