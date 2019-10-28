# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/10/28 14:52
# @author   :Mo
# @function :


# 适配linux
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_hyper_parameters
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_sim_webank_test
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessSim, load_json, extract_chinese
# 模型图
from keras_textclassification.m00_Bert.graph import BertGraph as Graph
# 模型评估
from sklearn.metrics import classification_report
# 计算时间
import time

import pandas as pd
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
    if path_test:  # 从外部引入测试数据地址
        hyper_parameters['data']['test_data'] = path_test
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    graph.load_model()
    print("graph load ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessSim()

    data = pd.read_csv(hyper_parameters['data']['test_data'])
    sentence_1 = data["sentence1"].values.tolist()
    sentence_2 = data["sentence2"].values.tolist()
    labels = data["label"].values.tolist()
    sentence_1 = [extract_chinese(str(line1).upper()) for line1 in sentence_1]
    sentence_2 = [extract_chinese(str(line2).upper()) for line2 in sentence_2]
    labels = [extract_chinese(str(line3).upper()) for line3 in labels]

    # 取该数据集的百分之几的语料测试
    len_rate = int(len(labels) * rate)
    sentence_1 = sentence_1[0:len_rate]
    sentence_2 = sentence_2[0:len_rate]
    labels = labels[0:len_rate]
    y_pred = []
    count = 0
    for i in range(len_rate):
        count += 1
        ques_embed = ra_ed.sentence2idx(text=sentence_1[i], second_text=sentence_2[i])
        if hyper_parameters['embedding_type'] in ['bert' or 'albert']:  # bert数据处理, token
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
            # 预测
            pred = graph.predict(x_val)
            pre = pt.prereocess_idx(pred[0])
            label_pred = pre[0][0][0]
            if count % 1000 == 0:
                print(label_pred)
            y_pred.append(label_pred)

    print("data pred ok!")
    # 预测结果转为int类型
    index_y = [pt.l2i_i2l['l2i'][i] for i in labels]
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
    pt = PreprocessSim()
    # 模式初始化和加载
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    sen1 = '我要打王者荣耀'
    sen2 = '我要打梦幻西游'

    # str to token
    ques_embed = ra_ed.sentence2idx(text=sen1, second_text=sen2)
    if hyper_parameters['embedding_type'] in ['bert', 'albert']:
        x_val_1 = np.array([ques_embed[0]])
        x_val_2 = np.array([ques_embed[1]])
        x_val = [x_val_1, x_val_2]
        # 预测
        pred = graph.predict(x_val)
        # 取id to label and pred
        pre = pt.prereocess_idx(pred[0])
        print(pre)
        while True:
            print("请输入sen1: ")
            sen1 = input()
            print("请输入sen2: ")
            sen2 = input()

            ques_embed = ra_ed.sentence2idx(text=sen1, second_text=sen2)
            print(ques_embed)
            if hyper_parameters['embedding_type'] in ['bert', 'albert']:
                x_val_1 = np.array([ques_embed[0]])
                x_val_2 = np.array([ques_embed[1]])
                x_val = [x_val_1, x_val_2]
                pred = graph.predict(x_val)
                pre = pt.prereocess_idx(pred[0])
                print(pre)
            else:
                print("error, just support bert or albert")

    else:
        print("error, just support bert or albert")


if __name__ == "__main__":
    # 测试集预测
    pred_tet(path_test=path_sim_webank_test, rate=1)  # sample条件下设为1,否则训练语料可能会很少

    # 可输入 input 预测
    pred_input()



