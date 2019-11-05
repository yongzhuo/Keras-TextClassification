# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/10/13 8:07
# @author   :Mo
# @function :数据切分为训练集,验证集


# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random

from keras_textclassification.data_preprocess.text_preprocess import txt_write, txt_read


def data_kfold(path_org_data, k_fold_split=10, path_save_dir=""):
    """
        切分训练-测试集, 使用sklearn的StratifiedKFold
    :param path_org_data: str, 原始语料绝对路径地址,utf-8的csv格式
    :param k_fold_split: int, k折切分, 原始语料中每个类至少有k_fold_split条句子
    :param path_save_dir: str, 生成训练集-测试集文件的保存目录
    :return: 
    """
    label_ques = pd.read_csv(path_org_data, names=["label","ques"], usecols=["label","ques"])
    quess = label_ques["ques"].values.tolist()[1:]
    labels = label_ques["label"].values.tolist()[1:]

    quess, labels = np.array(quess), np.array(labels)
    kf_sp = StratifiedKFold(n_splits=k_fold_split)

    for train_index, dev_index in kf_sp.split(quess, labels):
        train_x, train_y = quess[train_index], labels[train_index]
        dev_x, dev_y = quess[dev_index], labels[dev_index]
        print(len(set(train_y)))
        print(len(set(dev_y)))

        lq_train = [train_y[i].replace(",","，").strip() + "," + train_x[i].replace(",","，").strip() + "\n"
                    for i in range(len(train_y))]
        lq_valid = [dev_y[i].replace(",","，").strip() + "," + dev_x[i].replace(",","，").strip() + "\n"
                  for i in range(len(dev_y))]
        txt_write(["label,ques\n"] + lq_train, path_save_dir + "lq_train.csv")
        txt_write(["label,ques\n"] + lq_valid, path_save_dir + "lq_valid.csv")
        break



def data_split_train_val_label(path_org_data, path_save_dir, count_num=500000, use_shuffle=True):
    """
        解决numpy.array()报内存错误的情况,
        根据label划分训练-验证集, 保证每个类都选择
    :param path_org_data: str, 需要划分数据地址
    :param count_num:int, 预估一下数据量, 最好小于用例总量
    :return: 
    """
    def shuffle_corpus(corpus):
        # 先在label内shuffle
        random.shuffle(corpus)
        corpus_num = len(corpus)
        valid_portion = 0.2
        # +1,-1是为了保证该label下sample为1也可以取到
        train = corpus[0 : int((1 - valid_portion) * corpus_num) + 1]
        test = corpus[int((1 - valid_portion) * corpus_num) + 1-1 : ]
        return train, test
    # open().readline()单条数据读取
    datas = open(path_org_data, 'r', encoding='utf-8')
    data_all = {}
    label_set = set()
    count = 0
    while True:
        count += 1
        if count % 3200 ==0:
            print(count)
        line = datas.readline()
        # 跳出循环条件
        if not line and count > count_num:
            break
        if line:
            if line.strip() and count > 1:
                line_sp = line.strip().split(",")
                if len(line_sp) >= 2:
                    label = line_sp[0]
                    quest = line_sp[1]
                    if label == "":
                        label = "NAN"
                    label_set.add(label)
                    if label in data_all:
                        data_all[label].append(quest)
                    else:
                        data_all[label] = [quest]

    # 循环写入文件
    txt_write(['label,ques'+'\n'], path_save_dir + "train.csv")
    txt_write(['label,ques'+'\n'], path_save_dir + "valid.csv")
    for label_set_one in list(label_set):
        train, val = shuffle_corpus(data_all[label_set_one])
        train_ = [label_set_one + "," + t + "\n" for t in train]
        val_ = [label_set_one + "," + v + "\n" for v in val]

        txt_write(train_, path_save_dir + "train.csv", type='a+')
        txt_write(val_, path_save_dir + "valid.csv", type='a+')


    # 是否扰乱
    if use_shuffle:
        trains = txt_read("train.csv")
        valids = txt_read("valid.csv")
        random.shuffle(trains)
        random.shuffle(valids)
        trains = [t + "\n" for t in trains]
        valids = [v + "\n" for v in valids]
        txt_write(['label,ques'+'\n'], path_save_dir + "train.csv")
        txt_write(['label,ques'+'\n'], path_save_dir + "valid.csv")
        txt_write(trains, path_save_dir + "train.csv", type='a+')
        txt_write(valids, path_save_dir + "valid.csv", type='a+')



if __name__ == '__main__':

    from keras_textclassification.conf.path_config import path_root
    filepath = path_root + "/data/baidu_qa_2019/baike_qa_train.csv" # 原始语料
    k_fold_split = 10
    data_kfold(path_org_data=filepath, k_fold_split=10, path_save_dir=path_root+ "/data/baidu_qa_2019/")
    # data_split_train_val_label(path_org_data=filepath,
    #                            path_save_dir=path_root+ "/data/baidu_qa_2019/",
    #                            count_num = 500000, use_shuffle = True)
