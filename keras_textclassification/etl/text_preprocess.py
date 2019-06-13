# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:36
# @author   :Mo
# @function :data utils of text classification

from keras_textclassification.conf.path_config import path_fast_text_model
path_fast_text_model_vocab2index = path_fast_text_model + 'vocab2index.json'
path_fast_text_model_label2index = path_fast_text_model + 'label2index.json'

import pandas as pd
import numpy as np
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


def preprocess_baidu_qa_2019(path):
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
    #         x.append(line_x)
    #         y.append(line_y)
    # return x, y
    return x_y



def save_json(json_, path):
    """
      保存json，
    :param json_: json 
    :param path: str
    :return: None
    """
    with open(path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(json_))


def get_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.loads(fj.readlines()[0])
    return model_json


class PreprocessText:
    def __init__(self):
        gg = 0

    @staticmethod
    def prereocess_idx(pred):
        if os.path.exists(path_fast_text_model_label2index):
            pred_i2l = {}
            l2i_i2l = get_json(path_fast_text_model_label2index)
            i2l = l2i_i2l['i2l']
            for i in range(len(pred)):
                pred_i2l[i2l[str(i)]] = pred[i]
            pred_i2l_rank = [sorted(pred_i2l.items(), key=lambda k: k[1], reverse=True)]
            return pred_i2l_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_baidu_qa_2019_idx(self, path, embed):
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
        save_json(l2i_i2l, path_fast_text_model_label2index)

        x = []
        for que in ques:
            que_embed = embed.sentence2idx(que)
            x.append(que_embed)
        label_zo = []
        for label_one in label:
            label_zeros = [0] * len(l2i_i2l['l2i'])
            label_zeros[l2i_i2l['l2i'][label_one]] = 1
            label_zo.append(label_zeros)

        return np.array(x), np.array(label_zo)


if __name__=="__main__":
    # path = 'Y:/BaiduNetdiskDownload/DataSet/corpus/baike_qa2019/'
    # name = 'baike_qa_train.json'
    # # x, y = preprocess_baidu_qa_2019(path + name)
    # x_y = preprocess_baidu_qa_2019(path + name)
    # with open(name.replace('.json', '.csv'), 'w', encoding='utf-8') as f:
    #     f.writelines(x_y)

    from keras_textclassification.conf.path_config import path_baidu_qa_2019_valid
    pt = PreprocessText()
    pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid)