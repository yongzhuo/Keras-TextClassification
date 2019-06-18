# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of CharCNNGraph_kim with baidu-qa-2019 in question title
import os
import pathlib
import sys

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m03_CharCNN.graph_yoon_kim import CharCNNGraph as Graph

import random

if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 64,
                                     'embed_size': 30,
                                     'filters': [2, 3, 4], # 这里无用
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
                                     'embedding_type': 'random', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,
                                     'char_cnn_layers': [[50, 1], [100, 2], [150, 3],[200, 4], [200, 5], [200, 6],[200, 7]],  # large
                                     # [[25, 1], [50, 2], [75, 3], [100, 4], [125, 5], [150, 6]])  # small
                                     'highway_layers': 1, # large:2; small:1
                                     'num_rnn_layers': 1, # 论文是2，但训练实在是太慢了
                                     'rnn_type': 'GRU', # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                     'rnn_units': 650,  # large 650, small is 300
                                     'len_max_word': 26,
                                     },
                        'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,
                                      'level_type': 'char',
                                      'embed_size': 30,
                                      'len_max': 50,
                                      'len_max_word': 26
                                      },
                         }
    graph = Graph(hyper_parameters)
    ra_ed = graph.word_embedding
    pt = PreprocessText()
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed)
    indexs = [ids for ids in range(len(y_train))]
    random.shuffle(indexs)
    x_train, y_train = x_train[indexs], y_train[indexs]

    print(len(y_train))
    graph.fit(x_train[0:32000], y_train[0:32000], x_val, y_val)


# 1425170/1425170 [==============================] - 6498s 5ms/step - loss: 1.3809 - acc: 0.7042 - val_loss: 0.8345 - val_acc: 0.7534
# Epoch 00001: val_loss improved from inf to 0.83452, saving model to /home/ap/nlp/myzhuo/ClassificationTextChinese/data/model/fast_text/model_fast_text.f5
# Epoch 2/20
# 1425170/1425170 [==============================] - 6494s 5ms/step - loss: 0.8262 - acc: 0.7518 - val_loss: 0.7535 - val_acc: 0.7705
# Epoch 00002: val_loss improved from 0.83452 to 0.75352, saving model to /home/ap/nlp/myzhuo/ClassificationTextChinese/data/model/fast_text/model_fast_text.f5
# Epoch 3/20
#  306816/1425170 [=====>........................] - ETA: 1:23:55 - loss: 0.7666 - acc: 0.7673
