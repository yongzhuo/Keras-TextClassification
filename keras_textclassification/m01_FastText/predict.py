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
from keras_textclassification.m01_FastText.graph import FastTextGraph as Graph
# 模型评估
from sklearn.metrics import classification_report
# 计算时间
import time

import numpy as np


def pred_tet(path_hyper_parameter=path_hyper_parameters, path_test=None, rate=1.0):
    """
        测试集测试与模型评估
    :param hyper_parameters: json, 超参数
    :param path_test:str, path of test data, 测试集
    :param rate: 比率, 抽出rate比率语料取训练
    :return: None
    """
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
    """
       输入预测
    :param path_hyper_parameter: str, 超参存放地址
    :return: None
    """
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

# pred_input测试的result
# load_model start!
# load_model end!
# [[('游戏', 0.9869676), ('电脑', 0.0058066663), ('体育', 0.0014438167), ('社会', 0.0014223085), ('教育', 0.00073986716), ('娱乐', 0.0007245304), ('生活', 0.0006518028), ('文化', 0.0005466205), ('烦恼', 0.00050425093), ('商业', 0.00045706975), ('健康', 0.0002028785), ('汽车', 0.00013781844), ('电子', 0.00011276587), ('资源', 0.00010603126), ('NAN', 9.216716e-05), ('育儿', 7.4688454e-05), ('医疗', 9.102065e-06)]]
# 请输入:
# 你好呀
# [[101, 872, 1962, 1435, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# [[('娱乐', 0.23489292), ('烦恼', 0.17349072), ('教育', 0.13473377), ('健康', 0.086436085), ('游戏', 0.0777654), ('生活', 0.07014876), ('电脑', 0.06338945), ('社会', 0.04237141), ('文化', 0.033649065), ('体育', 0.030430038), ('商业', 0.019318668), ('育儿', 0.015810473), ('电子', 0.0052944664), ('资源', 0.0040681534), ('汽车', 0.003742728), ('NAN', 0.0030206428), ('医疗', 0.0014372484)]]
# 请输入:



# pred_tet测试,bert-char模式下1万4语料训练1epcoh后, 全量数据rate=0.01(448条语料),测试的样子
#               precision    recall  f1-score   support
#
#           电子  0.833333333 0.625000000 0.714285714         8
#           电脑  0.916666667 0.647058824 0.758620690        51
#           健康  0.812500000 0.639344262 0.715596330        61
#           烦恼  0.636363636 0.700000000 0.666666667        20
#           育儿  0.142857143 0.200000000 0.166666667         5
#           生活  0.553846154 0.734693878 0.631578947        49
#           文化  0.333333333 0.142857143 0.200000000         7
#           汽车  0.714285714 1.000000000 0.833333333         5
#           商业  0.722222222 0.742857143 0.732394366        35
#           教育  0.741379310 0.741379310 0.741379310        58
#           娱乐  0.696969697 0.575000000 0.630136986        40
#           体育  0.750000000 0.600000000 0.666666667         5
#           社会  0.500000000 0.500000000 0.500000000        12
#           游戏  0.792792793 0.956521739 0.866995074        92
#
#     accuracy                      0.720982143       448
#    macro avg  0.653325000 0.628908021 0.630308625       448
# weighted avg  0.732829132 0.720982143 0.718018262       448

