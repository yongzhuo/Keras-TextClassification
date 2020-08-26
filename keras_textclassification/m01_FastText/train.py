# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :train of fast text with baidu-qa-2019 in question title


# 适配linux
import pathlib
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 训练验证数据地址
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid, path_root

# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText, PreprocessSim, delete_file

# 模型图
from keras_textclassification.m01_FastText.graph import FastTextGraph as Graph
# from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph

# 计算时间
import time


# # fast_text config
# # 模型目录
# path_model_dir =  path_root + "/data/model/ccks_2020_el_cls_albert_fasttext/"
# # 语料地址
# path_model = path_root + '/data/model/ccks_2020_el_cls_albert_fasttext/model_fast_text.h5'
# # 超参数保存地址
# path_hyper_parameters =  path_root + '/data/model/ccks_2020_el_cls_albert_fasttext/hyper_parameters.json'
# # embedding微调保存地址
# path_fineture = path_root + "/data/model/ccks_2020_el_cls_albert_fasttext/embedding_trainable.h5"



# fast_text config
# # 模型目录
# path_model_dir =  path_root + "/data/model/ccks_2020_el_cls_random_fasttext/"
# # 语料地址
# path_model = path_root + '/data/model/ccks_2020_el_cls_random_fasttext/model_fast_text.h5'
# # 超参数保存地址
# path_hyper_parameters =  path_root + '/data/model/ccks_2020_el_cls_random_fasttext/hyper_parameters.json'
# # embedding微调保存地址
# path_fineture = path_root + "/data/model/ccks_2020_el_cls_random_fasttext/embedding_trainable.h5"
# if not os.path.exists(path_model_dir):
#     os.mkdir(path_model_dir)


def train(hyper_parameters=None, rate=1.0):
    if not hyper_parameters:
        hyper_parameters = {
        'len_max': 56,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
        'embed_size': 300,  # 字/词向量维度, bert取768, word取300, char可以更小些
        'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
        'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
        'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
        'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
        # 'gpu_memory_fraction': 0.76, #gpu使用率
        'model': {'label': 17,  # 类别数
                  'batch_size': 256,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                  'dropout': 0.5,  # 随机失活, 概率
                  'decay_step': 1000,  # 学习率衰减step, 每N个step衰减一次
                  'decay_rate': 0.999,  # 学习率衰减系数, 乘法
                  'filters': [3, 7, 7],
                  'filters_num': 300,  # 卷积个数 论文中 filters_num=150,300
                  'epochs': 20,  # 训练最大轮次
                  'patience': 3, # 早停,2-3就好
                  'lr': 1e-3,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                  'l2': 1e-9,  # l2正则化
                  'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数
                  'loss': 'categorical_crossentropy',  # 损失函数
                  'metrics': 'accuracy',  # 保存更好模型的评价标准
                  'optimizer_name': 'Adam', # 优化器, 可选['Adam', 'Radam', 'RAdam,Lookahead'], win10下必须使用GPU, 原因未知
                  'is_training': True,  # 训练后者是测试模型
                  'path_model_dir': path_model_dir, # 模型目录
                  'model_path': path_model,
                  # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                  'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                  'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                  },
        'embedding': {'layer_indexes': [24], # bert取的层数
                      # 'ngram_ns': [3],
                      # 'corpus_path': path_baidu_qa_2019_train,
                        },
        'data':{'train_data': path_baidu_qa_2019_train, # 训练数据
                'val_data': path_baidu_qa_2019_valid   # 验证数据
                },
    }

    # 删除先前存在的模型\embedding微调模型等
    delete_file(path_model_dir)
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessSim(path_model_dir)
    x_train, y_train = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                       hyper_parameters['data']['train_data'],
                                                       ra_ed, rate=rate, shuffle=True)
    x_val, y_val = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                   hyper_parameters['data']['val_data'],
                                                   ra_ed, rate=rate, shuffle=True)
    print("data propress ok!")
    print(len(y_train))
    # 训练
    graph.fit(x_train, y_train, x_val, y_val)
    # 训练
    # graph.fit_generator(embed=ra_ed, rate=rate)
    print("耗时:" + str(time.time()-time_start))


if __name__=="__main__":
    train(rate=1)
