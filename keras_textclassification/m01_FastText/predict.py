# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :train of fast text with baidu-qa-2019 in question title


import numpy as np

from keras_textclassification.conf.path_config import path_embedding_random_char
from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m01_FastText.graph import FastTextGraph

if __name__=="__main__":
    hyper_parameters = { 'model': {   'label': 17,
                                     'batch_size': 256,
                                     'embed_size': 300,
                                     'filters': [3, 4, 5],
                                     'kernel_size': 3,
                                     'channel_size': 1,
                                     'dropout': 0.5,
                                     'decay_step': 100,
                                     'decay_rate': 0.9,
                                     'epochs': 20,
                                     'len_max': 50,
                                     'vocab_size': 20000,
                                     'lr': 1e-4,
                                     'l2': 1e-9,
                                     'activate_classify': 'softmax',
                                     'embedding_type': 'random',
                                     'is_training': False,
                                     'model_path': path_model_fast_text_baiduqa_2019,},
                         'embedding':{ 'embedding_type': 'random',
                                      'corpus_path': path_embedding_random_char,
                                      'level_type': 'char',
                                      'embed_size': 300,
                                      'len_max': 50,},
                         }
    # ns = np.array([1,2,3,4])
    # print(type(ns))
    pt = PreprocessText
    graph = FastTextGraph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    ques = '你好呀'
    ques_embed = ra_ed.sentence2idx(ques)
    pred = graph.predict(np.array([ques_embed]))
    pre = pt.prereocess_idx(pred[0])
    print(pre)
    while True:
        print("请输入: ")
        ques = input()
        ques_embed = ra_ed.sentence2idx(ques)
        pred = graph.predict(np.array([ques_embed]))
        pre = pt.prereocess_idx(pred[0])
        print(pre)
