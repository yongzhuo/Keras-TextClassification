# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :train of textcnn-char-bert with baidu-qa-2019 in question title

import time

from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
from keras_textclassification.conf.path_config import path_embedding_bert
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph

import numpy as np


def list_to_numpy(x):
    """
        转换为bert能识别的
    :param x: 
    :return: 
    """
    x_1 = []
    x_2 = []
    for x in x_train:
        x_1.append(x[0])
        x_2.append(x[1])
    return [np.array(x_1), np.array(x_2)]

if __name__=="__main__":
    hyper_parameters = {'model': {   'label': 17,
                                     'batch_size': 256,
                                     'embed_size': 768,
                                     'filters': [3, 4, 5],
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
                                     'embedding_type': 'bert', # 还可以填'random'、 'bert' or 'word2vec"
                                     'is_training': True,
                                     'model_path': path_model_fast_text_baiduqa_2019,},
                        'embedding':{ 'embedding_type': 'bert',
                                      'corpus_path': path_embedding_bert,
                                      'level_type': 'char',
                                      'embed_size': 768,
                                      'len_max': 50,},
                         }

    time_start  = time.time()

    graph = Graph(hyper_parameters)
    ra_ed = graph.word_embedding
    ins = graph.word_embedding.input
    pt = PreprocessText()
    # rate= 0-1
    x_train, y_train = pt.preprocess_baidu_qa_2019_idx_bert(path_baidu_qa_2019_train, ra_ed, rate=1)
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx_bert(path_baidu_qa_2019_valid, ra_ed, rate=1)
    print(len(y_train))

    # x_train, y_train = x_train.tolist(), y_train.tolist()
    # x_val, y_val = x_val.tolist(), y_val.tolist()
    # x_train = list_to_numpy(x_train)
    # x_val = list_to_numpy(x_val)
    # y_train, y_val = np.array(y_train), np.array(y_val)

    graph.fit(x_train, y_train, x_val, y_val)
    print("耗时:" + str(time.time()-time_start))

term_char.txt
term_word.txt
w2v_model_merge_short.vec
w2v_model_wiki_char.vec