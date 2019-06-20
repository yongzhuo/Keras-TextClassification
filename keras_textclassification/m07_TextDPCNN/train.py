# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of DPCNN with baidu-qa-2019 in question title
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.m07_TextDPCNN.graph import DPCNNGraph as Graph
from keras_textclassification.etl.text_preprocess import PreprocessText
import random


if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 64,
                                     'embed_size': 300,
                                     'filters': 3, # 固定feature maps(filters)的数量
                                     'top_ks': [], # 这里无用
                                     'filters_num': 256,
                                     'channel_size': 1,
                                     'dropout': 0.5,
                                     'decay_step': 100,
                                     'decay_rate': 0.9,
                                     'epochs': 20,
                                     'len_max': 50,
                                     'vocab_size': 20000, #这里随便填的，会根据代码里修改
                                     'lr': 1e-3,
                                     'l2': 0.0000032,
                                     'activate_classify': 'softmax',
                                     'embedding_type': 'random', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,
                                     'rnn_type': 'GRU', # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                     'rnn_units': 650,  # large 650, small is 300
                                     'len_max_word': 26,
                                     # only DPCNN
                                     'pooling_size_strides': [3, 2], # 固定1/2池化
                                     'droupout_spatial': 0.2,
                                     'activation_conv': 'linear',    # Shortcut connections with pre-activation
                                     'layer_repeats': 7,
                                     'full_connect_unit': 256,
                                     'path_hyper_parameters': 'path_hyper_parameters.json'
                                     },
                        'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,
                                      'level_type': 'char',
                                      'embed_size': 300,
                                      'len_max': 50,
                                      'len_max_word': 26
                                      },
                         }

    import time
    time_start = time.time()

    graph = Graph(hyper_parameters)
    ra_ed = graph.word_embedding

    pt = PreprocessText()
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_train, ra_ed)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed)
    print(len(y_train))
    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time() - time_start))