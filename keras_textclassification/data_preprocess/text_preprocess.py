# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:36
# @author   :Mo
# @function :data utils of text classification


from keras_textclassification.conf.path_config import path_model_dir
path_fast_text_model_vocab2index = path_model_dir + 'vocab2index.json'
path_fast_text_model_l2i_i2l = path_model_dir + 'l2i_i2l.json'

import pandas as pd
import numpy as np
import random
import jieba
import json
import re
import os


def extract_chinese(text):
    """
      只提取出中文、字母和数字
    :param text: str, input of sentence
    :return: 
    """
    chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@.])", text))
    return chinese_exttract


def read_and_process(path):
    """
      读取文本数据并
    :param path: 
    :return: 
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_x = [extract_chinese(line.split(",")[0]) for line in lines]
        line_y = [extract_chinese(line.split(",")[1]) for line in lines]
        return lines_x, line_y


def preprocess_label_ques(path):
    x, y, x_y = [], [], []
    x_y.append('label,ques\n')
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            try:
                line_json = json.loads(line)
            except:
                break
            ques = line_json['title']
            label = line_json['category'][0:2]
            line_x = " ".join([extract_chinese(word) for word in list(jieba.cut(ques, cut_all=False, HMM=True))]).strip().replace('  ',' ')
            line_y = extract_chinese(label)
            x_y.append(line_y+','+line_x+'\n')
    return x_y


def save_json(jsons, json_path):
    """
      保存json，
    :param json_: json 
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(jsons))
    fj.close()


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.loads(fj.readlines()[0])
    return model_json


def delete_file(path):
    """
        删除一个目录下的所有文件
    :param path: str, dir path
    :return: None
    """
    for i in os.listdir(path):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path, i)
        if os.path.isfile(path_children):
            os.remove(path_children)
        else:# 递归, 删除目录下的所有文件
            delete_file(path_children)


class PreprocessText:
    """
        数据预处理, 输入为csv格式, [label,ques]
    """
    def __init__(self):
        self.l2i_i2l = None
        if os.path.exists(path_fast_text_model_l2i_i2l):
            self.l2i_i2l = load_json(path_fast_text_model_l2i_i2l)

    def prereocess_idx(self, pred):
        if os.path.exists(path_fast_text_model_l2i_i2l):
            pred_i2l = {}
            i2l = self.l2i_i2l['i2l']
            for i in range(len(pred)):
                pred_i2l[i2l[str(i)]] = pred[i]
            pred_i2l_rank = [sorted(pred_i2l.items(), key=lambda k: k[1], reverse=True)]
            return pred_i2l_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def prereocess_pred_xid(self, pred):
        if os.path.exists(path_fast_text_model_l2i_i2l):
            pred_l2i = {}
            l2i = self.l2i_i2l['l2i']
            for i in range(len(pred)):
                pred_l2i[pred[i]] = l2i[pred[i]]
            pred_l2i_rank = [sorted(pred_l2i.items(), key=lambda k: k[1], reverse=True)]
            return pred_l2i_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_label_ques_to_idx(self, embedding_type, path, embed, rate=1, shuffle=True):
        data = pd.read_csv(path)
        ques = data['ques'].tolist()
        label = data['label'].tolist()
        ques = [str(q).upper() for q in ques]
        label = [str(l).upper() for l in label]
        label_set = set(label)
        count = 0
        label2index = {}
        index2label = {}
        for label_one in label_set:
            label2index[label_one] = count
            index2label[count] = label_one
            count = count + 1
        l2i_i2l = {}
        l2i_i2l['l2i'] = label2index
        l2i_i2l['i2l'] = index2label
        save_json(l2i_i2l, path_fast_text_model_l2i_i2l)

        len_ql = int(rate * len(ques))
        if len_ql <= 5000: # sample时候不生效,使得语料足够训练
            len_ql = len(ques)

        x = []
        for que in ques[0:len_ql]:
            que_embed = embed.sentence2idx(que)
            x.append(que_embed) # [[], ]
        label_zo = []
        for label_one in label[0:len_ql]:
            label_zeros = [0] * len(l2i_i2l['l2i'])
            label_zeros[l2i_i2l['l2i'][label_one]] = 1
            label_zo.append(label_zeros)

        if embedding_type == 'bert':
            x_, y_ = np.array(x), np.array(label_zo)
            if shuffle:
                indexs = [ids for ids in range(len(y_))]
                random.shuffle(indexs)
                x_, y_ = x_[indexs], y_[indexs]
            x_1 = np.array([x[0] for x in x_])
            x_2 = np.array([x[1] for x in x_])
            x_all = [x_1, x_2]
            return x_all, y_
        else:
            x_, y_ = np.array(x), np.array(label_zo)
            if shuffle:
                indexs = [ids for ids in range(len(y_))]
                random.shuffle(indexs)
                x_, y_ = x_[indexs], y_[indexs]
            return x_, y_
