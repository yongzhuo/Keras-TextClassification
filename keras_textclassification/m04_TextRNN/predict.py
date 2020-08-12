# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :train of fast text with baidu-qa-2019 in question title

# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText, read_and_process, load_json
# 模型图
from keras_textclassification.m04_TextRNN.graph import TextRNNGraph as Graph
# 模型评估
from sklearn.metrics import classification_report
# 计算时间
import time

import numpy as np


def pred_tet(path_hyper_parameter=path_hyper_parameters, path_test=None, rate=1.0):
    # 测试集的准确率
    hyper_parameters = load_json(path_hyper_parameter)
    if path_test: # 从外部引入测试数据地址
        hyper_parameters['data']['val_data'] = path_test
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    graph.load_model()
    print("graph load ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessText(path_model_dir)
    y, x = read_and_process(hyper_parameters['data']['val_data'])
    # 取该数据集的百分之几的语料测试
    len_rate = int(len(y) * rate)
    x = x[1:len_rate]
    y = y[1:len_rate]
    y_pred = []
    count = 0
    for x_one in x:
        count += 1
        ques_embed = ra_ed.sentence2idx(x_one)
        if hyper_parameters['embedding_type'] in ['bert', 'albert']: # bert数据处理, token
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        # 预测
        pred = graph.predict(x_val)
        pre = pt.prereocess_idx(pred[0])
        label_pred = pre[0][0][0]
        if count % 1000==0:
            print(label_pred)
        y_pred.append(label_pred)

    print("data pred ok!")
    # 预测结果转为int类型
    index_y = [pt.l2i_i2l['l2i'][i] for i in y]
    index_pred = [pt.l2i_i2l['l2i'][i] for i in y_pred]
    target_names = [pt.l2i_i2l['i2l'][str(i)] for i in list(set((index_pred + index_y)))]
    # 评估
    report_predict = classification_report(index_y, index_pred,
                                           target_names=target_names, digits=9)
    print(report_predict)
    print("耗时:" + str(time.time() - time_start))


def pred_input(path_hyper_parameter=path_hyper_parameters):
    # 输入预测
    # 加载超参数
    hyper_parameters = load_json(path_hyper_parameter)
    pt = PreprocessText(path_model_dir)
    # 模式初始化和加载
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    ques = '我要打王者荣耀'
    # str to token
    ques_embed = ra_ed.sentence2idx(ques)
    if hyper_parameters['embedding_type'] in ['bert', 'albert']:
        x_val_1 = np.array([ques_embed[0]])
        x_val_2 = np.array([ques_embed[1]])
        x_val = [x_val_1, x_val_2]
    else:
        x_val = ques_embed
    # 预测
    pred = graph.predict(x_val)
    # 取id to label and pred
    pre = pt.prereocess_idx(pred[0])
    print(pre)
    while True:
        print("请输入: ")
        ques = input()
        ques_embed = ra_ed.sentence2idx(ques)
        print(ques_embed)
        if hyper_parameters['embedding_type'] in ['bert', 'albert']:
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        pred = graph.predict(x_val)
        pre = pt.prereocess_idx(pred[0])
        print(pre)


if __name__=="__main__":
    # 测试集预测
    pred_tet(path_test=path_baidu_qa_2019_valid, rate=1) # sample条件下设为1,否则训练语料可能会很少

    # 可输入 input 预测
    pred_input()

# rate=0.01,random,char下训练一轮预测结果
# 2019-07              precision    recall  f1-score   support
#
#           游戏  0.786407767 0.880434783 0.830769231        92
#           生活  0.471428571 0.673469388 0.554621849        49
#           社会  0.363636364 0.333333333 0.347826087        12
#           文化  0.000000000 0.000000000 0.000000000         7
#           健康  0.775862069 0.737704918 0.756302521        61
#           体育  0.000000000 0.000000000 0.000000000         5
#           娱乐  0.549019608 0.700000000 0.615384615        40
#           商业  0.863636364 0.542857143 0.666666667        35
#           烦恼  0.722222222 0.650000000 0.684210526        20
#           汽车  0.000000000 0.000000000 0.000000000         5
#           电脑  0.722222222 0.764705882 0.742857143        51
#           教育  0.650000000 0.672413793 0.661016949        58
#           育儿  0.000000000 0.000000000 0.000000000         5
#           电子  0.000000000 0.000000000 0.000000000         8
#
#     accuracy                      0.671875000       448
#    macro avg  0.421745370 0.425351374 0.418546828       448
# weighted avg  0.643541455 0.671875000 0.651279537       448
