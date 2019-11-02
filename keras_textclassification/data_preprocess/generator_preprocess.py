# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/2 21:08
# @author  : Mo
# @function:


from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json
from keras_textclassification.conf.path_config import path_model_dir
path_fast_text_model_vocab2index = path_model_dir + 'vocab2index.json'
path_fast_text_model_l2i_i2l = path_model_dir + 'l2i_i2l.json'

import numpy as np
import os


class PreprocessGenerator:
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

    def preprocess_get_label_set(self, path):
        # 首先获取label,set,即存在的具体类
        label_set = set()
        len_all = 0
        file_csv = open(path, "r", encoding="utf-8")
        for line in file_csv:
            len_all += 1
            if len_all > 1:  # 第一条是标签'label,ques'，不选择
                line_sp = line.split(",")
                label_org = str(line_sp[0]).strip().upper()
                label_real = "NAN" if label_org=="" else label_org
                label_set.add(label_real)
        file_csv.close()
        return label_set, len_all

    def preprocess_label_ques_to_idx(self, embedding_type, batch_size, path, embed, rate=1):
        label_set, len_all = self.preprocess_get_label_set(path)
        # 获取label转index字典等, 如果label2index存在则不转换了, dev验证集合的时候用
        if not os.path.exists(path_fast_text_model_l2i_i2l):
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
        else:
            l2i_i2l = load_json(path_fast_text_model_l2i_i2l)

        # 读取数据的比例
        len_ql = int(rate * len_all)
        if len_ql <= 500:  # sample时候不生效,使得语料足够训练
            len_ql = len_all

        def process_line(line):
            # 对每一条数据操作，获取label和问句index
            line_sp = line.split(",")
            ques = str(line_sp[1]).strip().upper()
            label = str(line_sp[0]).strip().upper()
            label = "NAN" if label == "" else label
            que_embed = embed.sentence2idx(ques)
            label_zeros = [0] * len(l2i_i2l['l2i'])
            label_zeros[l2i_i2l['l2i'][label]] = 1
            return que_embed, label_zeros

        while True:
            file_csv = open(path, "r", encoding="utf-8")
            cout_all_line = 0
            cnt = 0
            x, y = [], []
            # 跳出循环
            if len_ql < cout_all_line:
                break
            for line in file_csv:
                cout_all_line += 1
                if cout_all_line > 1: # 第一条是标签'label,ques'，不选择
                    x_line, y_line = process_line(line)
                    x.append(x_line)
                    y.append(y_line)
                    cnt += 1
                    if cnt == batch_size:
                        if embedding_type in ['bert', 'albert']:
                            x_, y_ = np.array(x), np.array(y)
                            x_1 = np.array([x[0] for x in x_])
                            x_2 = np.array([x[1] for x in x_])
                            x_all = [x_1, x_2]
                        elif embedding_type == 'xlnet':
                            x_, y_ = x, np.array(y)
                            x_1 = np.array([x[0][0] for x in x_])
                            x_2 = np.array([x[1][0] for x in x_])
                            x_3 = np.array([x[2][0] for x in x_])
                            x_all = [x_1, x_2, x_3]
                        else:
                            x_all, y_ = np.array(x), np.array(y)

                        cnt = 0
                        yield (x_all, y_)
                        x, y =[], []
        file_csv.close()
        print("preprocess_label_ques_to_idx ok")




