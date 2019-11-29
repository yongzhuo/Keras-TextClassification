# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/12 16:45
# @author  : Mo
# @function:


from keras_textclassification import train
train(graph='TextCNN', # 必填, 算法名, 可选"ALBERT","BERT","XLNET","FASTTEXT","TEXTCNN","CHARCNN",
                       # "TEXTRNN","RCNN","DCNN","DPCNN","VDCNN","CRNN","DEEPMOJI",
                       # "SELFATTENTION", "HAN","CAPSULE","TRANSFORMER"
     label=17,         # 必填, 类别数, 训练集和测试集合必须一样
     path_train_data=None, # 必填, 训练数据文件, csv格式, 必须含'label,ques'头文件, 详见keras_textclassification/data
     path_dev_data=None, # 必填, 测试数据文件, csv格式, 必须含'label,ques'头文件, 详见keras_textclassification/data
     rate=1,             # 可填, 训练数据选取比例
     hyper_parameters=None) # 可填, json格式, 超参数, 默认embedding为'char','random'
