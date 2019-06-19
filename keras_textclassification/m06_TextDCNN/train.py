# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of CharCNNGraph_kim with baidu-qa-2019 in question title
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_hyper_parameters_fast_text
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.m06_TextDCNN.graph import DCNNGraph as Graph
from keras_textclassification.etl.text_preprocess import PreprocessText
import random


if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 64,
                                     'embed_size': 300,
                                     'filters': [[10, 7, 5], [6, 4, 3]], # 3层的时候
                                     # 'filters': [[10, 7], [5, 3]],  # 2层的时候
                                     # 'filters': [[5, 3], [4, 2]], #2层的时候
                                     # 'top_ks': [[6, 3], [7, 4], [9, 3]], # 这里无用
                                     'filters_num': 300,
                                     'channel_size': 1,
                                     'dropout': 0.5,
                                     'decay_step': 100,
                                     'decay_rate': 0.9,
                                     'epochs': 20,
                                     'len_max': 30,
                                     'vocab_size': 20000,  #这里随便填的，会根据代码里修改
                                     'lr': 1e-3,
                                     'l2': 1e-6,
                                     'activate_classify': 'softmax',
                                     'embedding_type': 'random',  # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,
                                     'path_hyper_parameters': path_hyper_parameters_fast_text,
                                     'rnn_type': 'GRU',  # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                     'rnn_units': 650,  # large 650, small is 300
                                     'len_max_word': 26,
                                     },
                        'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,
                                      'level_type': 'char',
                                      'embed_size': 300,
                                      'len_max': 30,
                                      'len_max_word': 26
                                      },
                         }
    # graph = Graph(hyper_parameters)
    # ra_ed = graph.word_embedding
    # pt = PreprocessText()
    # x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed)
    # x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed)
    # indexs = [ids for ids in range(len(y_train))]
    # random.shuffle(indexs)
    # x_train, y_train = x_train[indexs].tolist(), y_train[indexs].tolist()
    # x_val, y_val = x_val.tolist(), y_val.tolist()
    # print(len(y_train))
    # graph.fit(x_train, y_train, x_val, y_val)
    import time
    time_start = time.time()

    graph = Graph(hyper_parameters)
    ra_ed = graph.word_embedding
    pt = PreprocessText()
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed, rate=0.1)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed, rate=0.1)
    indexs = [ids for ids in range(len(y_train))]
    random.shuffle(indexs)
    x_train, y_train = x_train[indexs], y_train[indexs]

    print(len(y_train))
    # 只训练部分
    # graph.fit(x_train[0:32000], y_train[0:32000], x_val[0:3200], y_val[0:3200])

    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time() - time_start))