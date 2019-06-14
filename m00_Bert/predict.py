# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/12 14:11
# @author   :Mo
# @function :


from keras_textclassification.conf.path_config import path_model_fast_text_baiduqa_2019
from keras_textclassification.conf.path_config import path_embedding_bert
from keras_textclassification.etl.text_preprocess import PreprocessText
from keras_textclassification.m00_Bert.graph import BertGraph as Graph
import numpy as np


def pred_tet():
    # 测试集的准确率
    from keras_textclassification.conf.path_config import path_baidu_qa_2019_valid
    hyper_parameters = {'model': {'label': 17,
                                  'batch_size': 64,
                                  'embed_size': 30,
                                  'filters': [2, 3, 4],
                                  'kernel_size': 30,
                                  'channel_size': 1,
                                  'dropout': 0.5,
                                  'decay_step': 100,
                                  'decay_rate': 0.9,
                                  'epochs': 20,
                                  'len_max': 50,
                                  'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
                                  'lr': 1e-3,
                                  'l2': 1e-6,
                                  'activate_classify': 'softmax',
                                  'embedding_type': 'bert',  # 还可以填'random'、 'bert' or 'word2vec"
                                  'is_training': True,
                                  'model_path': path_model_fast_text_baiduqa_2019,  # 地址可以自己设置
                                  'rnn_type': 'GRU',
                                  # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                  'rnn_units': 256,  # large 650, small is 300
                                  },
                        'embedding': {'embedding_type': 'bert',
                                      'corpus_path': path_embedding_bert,
                                      'level_type': 'char',
                                      'embed_size': 30,
                                      'len_max': 50,
                                      'layer_indexes': [12]  # range 1 to 12
                                      },
                        }
    pt = PreprocessText()
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    x_val, y_val = pt.preprocess_baidu_qa_2019_idx(path_baidu_qa_2019_valid, ra_ed, rate=1)
    x_val = x_val.tolist()
    y_val = y_val.tolist()
    y_pred = []
    count = 0
    for x_val_one in x_val:
        count = count + 1
        print(x_val_one)
        ques_embed_1 = np.array([x_val_one[0]])
        ques_embed_2 = np.array([x_val_one[1]])
        pred = graph.predict([ques_embed_1, ques_embed_2])
        print(pred)
        pred_top1 = pt.prereocess_pred_id(pred[0])
        print(pred_top1)
        y_pred.append(pred_top1)

    acc = 0
    for i in range(len(y_val)):
        if y_val[i] == y_pred[i]:
            acc += 1
    print('true: {}  total: {}  acc: {}'.format(acc, len(y_val), acc/len(y_val)))




def pred_input():
    hyper_parameters = {'model': {'label': 17,
                                  'batch_size': 64,
                                  'embed_size': 30,
                                  'filters': [2, 3, 4],
                                  'kernel_size': 30,
                                  'channel_size': 1,
                                  'dropout': 0.5,
                                  'decay_step': 100,
                                  'decay_rate': 0.9,
                                  'epochs': 20,
                                  'len_max': 50,
                                  'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
                                  'lr': 1e-3,
                                  'l2': 1e-6,
                                  'activate_classify': 'softmax',
                                  'embedding_type': 'bert',  # 还可以填'random'、 'bert' or 'word2vec"
                                  'is_training': True,
                                  'model_path': path_model_fast_text_baiduqa_2019,  # 地址可以自己设置
                                  'rnn_type': 'GRU',
                                  # type of rnn, select 'LSTM', 'GRU', 'CuDNNGRU', 'CuDNNLSTM', 'Bidirectional-LSTM', 'Bidirectional-GRU'
                                  'rnn_units': 256,  # large 650, small is 300
                                  },
                        'embedding': {'embedding_type': 'bert',
                                      'corpus_path': path_embedding_bert,
                                      'level_type': 'char',
                                      'embed_size': 30,
                                      'len_max': 50,
                                      'layer_indexes': [12]  # range 1 to 12
                                      },
                        }

    pt = PreprocessText()
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    ques = '我要打王者荣耀'
    ques_embed = ra_ed.sentence2idx(ques)
    x_val_1 = np.array([ques_embed[0]])
    x_val_2 = np.array([ques_embed[1]])
    x_val = [x_val_1, x_val_2]
    pred = graph.predict(x_val)
    pre = pt.prereocess_idx(pred[0])
    print(pre)
    while True:
        print("请输入: ")
        ques = input()
        ques_embed = ra_ed.sentence2idx(ques)
        print(ques_embed)
        ques_embed_1 = np.array([ques_embed[0]])
        ques_embed_2 = np.array([ques_embed[1]])
        pred = graph.predict([ques_embed_1, ques_embed_2])
        pre = pt.prereocess_idx(pred[0])
        print(pre)


if __name__=="__main__":
    # 可输入 input 预测
    # pred_input()

    # 测试集预测
    pred_tet()


