# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/8 10:14
# @author  : Mo
# @function:


# linux适配
import pathlib
import sys
import os

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import delete_file
# 模型图
from keras_textclassification.m00_Albert.graph import AlbertGraph
from keras_textclassification.m00_Bert.graph import BertGraph
from keras_textclassification.m00_Xlnet.graph import XlnetGraph
from keras_textclassification.m01_FastText.graph import FastTextGraph
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph
from keras_textclassification.m03_CharCNN.graph_yoon_kim import CharCNNGraph
from keras_textclassification.m04_TextRNN.graph import TextRNNGraph
from keras_textclassification.m05_TextRCNN.graph import RCNNGraph
from keras_textclassification.m06_TextDCNN.graph import DCNNGraph
from keras_textclassification.m07_TextDPCNN.graph import DPCNNGraph
from keras_textclassification.m08_TextVDCNN.graph import VDCNNGraph
from keras_textclassification.m09_TextCRNN.graph import CRNNGraph
from keras_textclassification.m10_DeepMoji.graph import DeepMojiGraph
from keras_textclassification.m11_SelfAttention.graph import SelfAttentionGraph
from keras_textclassification.m12_HAN.graph import HANGraph
from keras_textclassification.m13_CapsuleNet.graph import CapsuleNetGraph
from keras_textclassification.m14_Transformer.graph import TransformerEncodeGraph
# 计算时间
import time


def train(graph='TextCNN', label=17, rate=1.0, hyper_parameters=None, path_train_data=None, path_dev_data=None):
    """
    
        训练函数
    :param hyper_parameters: json, 超参数
    :param rate: float, 比率, 抽出rate比率语料取训练
    :param graph: str, 具体模型
    :param path_train_data: str, 训练数据集地址
    :param path_dev_data: str, 验证数据集地址
    :return: None
    """

    # 模型选择
    str2graph = {"ALBERT": AlbertGraph,
                 "BERT": BertGraph,
                 "XLNET": XlnetGraph,
                 "FASTTEXT": FastTextGraph,
                 "TEXTCNN": TextCNNGraph,
                 "CHARCNN": CharCNNGraph,
                 "TEXTRNN": TextRNNGraph,
                 "RCNN": RCNNGraph,
                 "DCNN": DCNNGraph,
                 "DPCNN": DPCNNGraph,
                 "VDCNN": VDCNNGraph,
                 "CRNN": CRNNGraph,
                 "DEEPMOJI": DeepMojiGraph,
                 "SELFATTENTION": SelfAttentionGraph,
                 "HAN": HANGraph,
                 "CAPSULE": CapsuleNetGraph,
                 "TRANSFORMER": TransformerEncodeGraph
                 }
    graph = graph.upper()
    Graph = str2graph[graph] if graph in str2graph.keys() else str2graph["TEXTCNN"]
    hyper_parameters_real = {
        'len_max': 50,  # 句子最大长度, 固定 推荐20-50
        'trainable': True,  # embedding是静态的还是动态的
        'embed_size': 64,  # 字/词向量维度
        'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
        'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word'
        'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
        'gpu_memory_fraction': 0.66,  # gpu使用率
        'model': {'label': label,  # 类别数
                  'batch_size': 64,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                  'dropout': 0.1,  # 随机失活, 概率
                  'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
                  'decay_rate': 0.9,  # 学习率衰减系数, 乘法
                  'epochs': 50,  # 训练最大轮次
                  'patience': 5,  # 早停,2-3就好
                  'lr': 1e-3,  # 学习率, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                  'l2': 1e-9,  # l2正则化
                  'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数
                  'loss': 'categorical_crossentropy',  # 损失函数
                  'metrics': 'accuracy',  # 保存更好模型的评价标准
                  'is_training': True,  # 训练后者是测试模型
                  'model_path': path_model,  # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                  'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                  'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                  'droupout_spatial': 0.25,
                  'encoder_num': 1,
                  'head_num': 12,
                  'hidden_dim': 3072,
                  'attention_activation': 'relu',
                  'feed_forward_activation': 'relu',
                  'use_adapter': False,
                  'adapter_units': 768,
                  'adapter_activation': 'relu',
                  },
        'embedding': {'layer_indexes': [12],  # bert取的层数,
                      # 'corpus_path': '',     # embedding预训练数据地址,不配则会默认取conf里边默认的地址, keras-bert可以加载谷歌版bert,百度版ernie(需转换，https://github.com/ArthurRizar/tensorflow_ernie),哈工大版bert-wwm(tf框架，https://github.com/ymcui/Chinese-BERT-wwm)
                      },
        'data': {'train_data': path_train_data if path_train_data else path_baidu_qa_2019_train,  # 训练数据
                 'val_data': path_dev_data if path_dev_data else path_baidu_qa_2019_valid  # 验证数据
                 },
    }

    # 自定义超参数的
    if hyper_parameters:
        hyper_gpu_memory_fraction = hyper_parameters.get('gpu_memory_fraction', {})
        hyper_embedding_type = hyper_parameters.get('embedding_type', {})
        hyper_embed_size = hyper_parameters.get('embed_size', {})
        hyper_level_type = hyper_parameters.get('level_type', {})
        hyper_trainable = hyper_parameters.get('trainable', {})
        hyper_len_max = hyper_parameters.get('len_max', {})

        hyper_parameters_real['gpu_memory_fraction'] = hyper_gpu_memory_fraction if hyper_gpu_memory_fraction else hyper_parameters_real['gpu_memory_fraction']
        hyper_parameters_real['embedding_type'] = hyper_embedding_type if hyper_embedding_type else hyper_parameters_real['embedding_type']
        hyper_parameters_real['level_type'] = hyper_level_type if hyper_level_type else hyper_parameters_real['level_type']
        hyper_parameters_real['embed_size'] = hyper_embed_size if hyper_embed_size else hyper_parameters_real['embed_size']
        hyper_parameters_real['trainable'] = hyper_trainable if hyper_trainable else hyper_parameters_real['trainable']
        hyper_parameters_real['len_max'] = hyper_len_max if hyper_len_max else hyper_parameters_real['len_max']

        hyper_model = hyper_parameters.get('model', {})
        hyper_embedding = hyper_parameters.get('embedding', {})
        hyper_data = hyper_parameters.get('data', {})
        for hm in hyper_model.keys():
            hyper_parameters_real[hm] = hyper_model[hm]
        for he in hyper_embedding.keys():
            hyper_parameters_real[he] = hyper_model[he]
        for hd in hyper_embedding.keys():
            hyper_parameters_real[hd] = hyper_data[hd]

        # 选择bert及其改进模型的, lr
        if hyper_parameters_real['embedding_type'] in ['xlnet', 'bert', 'albert']:
            if hyper_parameters_real['model']['lr'] > 1e-4:
                hyper_parameters_real['model']['lr'] = 1e-5

    # 删除先前存在的模型和embedding微调模型等
    delete_file(path_model_dir)
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters_real)
    print("graph init ok!")
    ra_ed = graph.word_embedding

    # 训练
    graph.fit_generator(embed=ra_ed, rate=rate)
    print("耗时:" + str(time.time() - time_start))


if __name__ == "__main__":
    train(graph='TextCNN', label=17, rate=1, path_train_data=None, path_dev_data=None,hyper_parameters=None)
