# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/10/13 8:07
# @author   :Mo
# @function :数据切分为训练集,验证集


from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

from keras_textclassification.data_preprocess.text_preprocess import txt_write


def data_kfold(path_org_data, k_fold_split=10, path_save_dir=""):
    """
        切分训练-测试集
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
        lq_train = [train_y[i].replace(",","，").strip() + "," + train_x[i].replace(",","，").strip() + "\n"
                    for i in range(len(train_y))]
        lq_valid = [dev_y[i].replace(",","，").strip() + "," + dev_x[i].replace(",","，").strip() + "\n"
                  for i in range(len(dev_y))]
        txt_write(["label,ques\n"] + lq_train, path_save_dir + "lq_train.csv")
        txt_write(["label,ques\n"] + lq_valid, path_save_dir + "lq_valid.csv")
        break

if __name__ == '__main__':

    from keras_textclassification.conf.path_config import path_root
    filepath = path_root + "/data/baidu_qa_2019/baike_qa_train.csv" # 原始语料
    k_fold_split = 10
    data_kfold(path_org_data=filepath, k_fold_split=10, path_save_dir=path_root+ "/data/baidu_qa_2019/")