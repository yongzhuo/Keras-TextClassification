# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/8 14:37
# @author   :Mo
# @function :train of VDCNN with baidu-qa-2019 in question title
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_hyper_parameters_fast_text, path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.m08_TextVDCNN.graph import VDCNNGraph as Graph
from keras_textclassification.etl.text_preprocess import PreprocessText
import random


if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 64,
                                     'embed_size': 300,
                                     # only VDCNN
                                     'top_k': 2,
                                     'pool_type': 'max',
                                     'shortcut': True,
                                     # 论文参数, 长文本, long sentence, ,max_len=256 or 1024 ans so on
                                     'filters': [[64, 1],[128, 1],[256, 1],[512, 1]], # 9 layer, 256 len max
                                     # 'filters': [[64, 2], [128, 2], [256, 2], [512, 2]],  # 17 layer
                                     # 'filters': [[64, 5], [128, 5], [256, 2], [512, 2]],  # 29 layer
                                     # 'filters': [[64, 8], [128, 8], [256, 5], [512, 3]],  # 49 layer, 1024 len max

                                     # 自己设置的,效果不太佳, 短文本, short sentence, ,max_len=32 or 64 ans so on
                                     # 'filters': [[3, 8], [6, 8], [12, 5], [24, 3]],  # 49 layer
                                     # 'filters': [[4, 8], [8, 8], [16, 5], [32, 3]],  # 49 layer
                                     # 'filters': [[3, 1], [6, 1], [12, 1], [24, 1]],  # 9 layer
                                     # 'filters': [[4, 1], [8, 1], [16, 1], [32, 1]],  # 9 layer
                                     'channel_size': 1,
                                     'dropout': 0.5,
                                     'decay_step': 100,
                                     'decay_rate': 0.9,
                                     'epochs': 20,
                                     'len_max': 256, # layer 9 就需要256的len_max，只适合长文本，如新闻等
                                     'vocab_size': 20000, #这里随便填的，会根据代码里修改
                                     'lr': 1e-3,
                                     'l2': 0.0000032,
                                     'activate_classify': 'softmax',
                                     'embedding_type': 'random', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,
                                     'path_hyper_parameters': path_hyper_parameters_fast_text,
                                     },
                        'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,
                                      'level_type': 'char',
                                      'embed_size': 300,
                                      'len_max': 256, # 1024(filters:64, layer:49), 256(filters:64, layer:9), 64(filters:3, layer:9)
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