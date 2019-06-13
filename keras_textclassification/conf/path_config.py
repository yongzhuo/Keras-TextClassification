# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:04
# @author   :Mo
# @function :file of path

import os

# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# path of embedding
path_embedding_random_char = path_root + '/data/embeddings/term_char.txt'
path_embedding_bert = path_root + '/data/enbeddings/bert/'
path_embedding_vector_word2vec = path_root + '/data/embeddings/'

# classify data of baidu qa 2019
path_baidu_qa_2019_train = path_root + '/data/baidu_qa_2019/baike_qa_train.csv'
path_baidu_qa_2019_valid = path_root + '/data/baidu_qa_2019/baike_qa_valid.csv'

# fast_text config
path_fast_text_model = path_root + '/data/model/fast_text/'
path_model_fast_text_baiduqa_2019 = path_root + '/data/model/fast_text/model_fast_text.f5'
