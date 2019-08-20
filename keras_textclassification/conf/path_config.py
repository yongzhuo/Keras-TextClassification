# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:04
# @author   :Mo
# @function :file of path

import os

# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_root = path_root.replace('\\', '/')

# path of embedding
path_embedding_random_char = path_root + '/data/embeddings/term_char.txt'
path_embedding_random_word = path_root + '/data/embeddings/term_word.txt'
path_embedding_bert = path_root + '/data/embeddings/chinese_L-12_H-768_A-12/'
path_embedding_vector_word2vec_char = path_root + '/data/embeddings/w2v_model_wiki_char.vec'
path_embedding_vector_word2vec_word = path_root + '/data/embeddings/w2v_model_merge_short.vec'

# classify data of baidu qa 2019
path_baidu_qa_2019_train = path_root + '/data/baidu_qa_2019/baike_qa_train.csv'
path_baidu_qa_2019_valid = path_root + '/data/baidu_qa_2019/baike_qa_valid.csv'

# 今日头条新闻多标签分类
path_byte_multi_news_train = path_root + '/data/byte_multi_news/train.csv'
path_byte_multi_news_valid = path_root + '/data/byte_multi_news/valid.csv'
path_byte_multi_news_label = path_root + '/data/byte_multi_news/labels.csv'

# fast_text config
# 模型目录
path_model_dir =  path_root + "/data/model/fast_text/"
# 语料地址
path_model = path_root + '/data/model/fast_text/model_fast_text.h5'
# 超参数保存地址
path_hyper_parameters =  path_root + '/data/model/fast_text/hyper_parameters.json'
# embedding微调保存地址
path_fineture = path_root + "/data/model/fast_text/embedding_trainable.h5"
