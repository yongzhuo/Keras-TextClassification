# [Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification)

[![PyPI](https://img.shields.io/pypi/v/Keras-TextClassification)](https://pypi.org/project/Keras-TextClassification/)
[![Build Status](https://travis-ci.com/yongzhuo/Keras-TextClassification.svg?branch=master)](https://travis-ci.com/yongzhuo/Keras-TextClassification)
[![PyPI_downloads](https://img.shields.io/pypi/dm/Keras-TextClassification)](https://pypi.org/project/Keras-TextClassification/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/Keras-TextClassification?style=social)](https://github.com/yongzhuo/Keras-TextClassification/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/Keras-TextClassification.svg?style=social)](https://github.com/yongzhuo/Keras-TextClassification/network/members)
[![Join the chat at https://gitter.im/yongzhuo/Keras-TextClassification](https://badges.gitter.im/yongzhuo/Keras-TextClassification.svg)](https://gitter.im/yongzhuo/Keras-TextClassification?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)



# Install(安装)

```bash
pip install Keras-TextClassification
```

```python
step2: download and unzip the dir of 'data.rar', 地址: https://pan.baidu.com/s/1I3vydhmFEQ9nuPG2fDou8Q 提取码: rket
       cover the dir of data to anaconda, like '/anaconda/3.5.1/envs/tensorflow13/Lib/site-packages/keras_textclassification/data'
step3: goto # Train&Usage(调用) and Predict&Usage(调用)
```

# keras_textclassification（代码主体,未完待续...）
    - Albert-fineture
    - Xlnet-fineture
    - Bert-fineture
    - FastText
    - TextCNN
    - charCNN
    - TextRNN
    - TextRCNN
    - TextDCNN
    - TextDPCNN
    - TextVDCNN
    - TextCRNN
    - DeepMoji
    - SelfAttention
    - HAN
    - CapsuleNet
    - Transformer-encode


# run(运行, 以FastText为例)
    - 1. 进入keras_textclassification/m01_FastText目录，
    - 2. 训练: 运行 train.py,   例如: python train.py
    - 3. 预测: 运行 predict.py, 例如: python predict.py
    - 说明: 默认不带pre train的random embedding，训练和验证语料只有100条，完整语料移步下面data查看下载

# run(多标签分类/Embedding/test/sample实例)
    - bert,word2vec,random样例在test/目录下, 注意word2vec(char or word), random-word,  bert(chinese_L-12_H-768_A-12)未全部加载,需要下载
    - multi_multi_class/目录下以text-cnn为例进行多标签分类实例，转化为multi-onehot标签类别，分类则取一定阀值的类
    - sentence_similarity/目录下以bert为例进行两个句子文本相似度计算,数据格式如data/sim_webank/目录下所示
    - predict_bert_text_cnn.py
    - tet_char_bert_embedding.py
    - tet_char_bert_embedding.py
    - tet_char_xlnet_embedding.py
    - tet_char_random_embedding.py
    - tet_char_word2vec_embedding.py
    - tet_word_random_embedding.py
    - tet_word_word2vec_embedding.py

# keras_textclassification/data
    - 数据下载
      ** github项目中只是上传部分数据，需要的前往链接: https://pan.baidu.com/s/1I3vydhmFEQ9nuPG2fDou8Q 提取码: rket
    - baidu_qa_2019（百度qa问答语料，只取title作为分类样本，17个类，有一个是空''，已经压缩上传）
       - baike_qa_train.csv
       - baike_qa_valid.csv
    - byte_multi_news（今日头条2018新闻标题多标签语料，1070个标签，fate233爬取, 地址为: [byte_multi_news](https://github.com/fate233/toutiao-multilevel-text-classfication-dataset)）
       -labels.csv
       -train.csv
       -valid.csv
    - embeddings
       - chinese_L-12_H-768_A-12/(取谷歌预训练好点的模型,已经压缩上传,
                                  keras-bert还可以加载百度版ernie(需转换，[https://github.com/ArthurRizar/tensorflow_ernie](https://github.com/ArthurRizar/tensorflow_ernie)),
                                  哈工大版bert-wwm(tf框架，[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm))
       - albert_base_zh/(brightmart训练的albert, 地址为https://github.com/brightmart/albert_zh)
       - chinese_xlnet_mid_L-24_H-768_A-12/(哈工大预训练的中文xlnet模型[https://github.com/ymcui/Chinese-PreTrained-XLNet],24层)
       - term_char.txt(已经上传, 项目中已全, wiki字典, 还可以用新华字典什么的)
       - term_word.txt(未上传, 项目中只有部分, 可参考词向量的)
       - w2v_model_merge_short.vec(未上传, 项目中只有部分, 词向量, 可以用自己的)
       - w2v_model_wiki_char.vec(已上传百度网盘, 项目中只有部分, 自己训练的维基百科字向量, 可以用自己的)
    - model
       - fast_text/预训练模型存放地址

# 项目说明
  - 1. 构建了base基类(网络(graph)、向量嵌入(词、字、句子embedding)),后边的具体模型继承它们，代码简单
  - 2. keras_layers存放一些常用的layer, conf存放项目数据、模型的地址, data存放数据和语料, data_preprocess为数据预处理模块,


# 模型与论文paper题与地址
* FastText:   [Bag of Tricks for Efﬁcient Text Classiﬁcation](https://arxiv.org/abs/1607.01759)
* TextCNN：   [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)
* charCNN-kim：   [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
* charCNN-zhang:  [Character-level Convolutional Networks for Text Classiﬁcation](https://arxiv.org/pdf/1509.01626.pdf)
* TextRNN：   [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
* RCNN：      [Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
* DCNN:       [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188)
* DPCNN:      [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://www.aclweb.org/anthology/P17-1052)
* VDCNN:      [Very Deep Convolutional Networks](https://www.aclweb.org/anthology/E17-1104)
* CRNN:        [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)
* DeepMoji:    [Using millions of emojio ccurrences to learn any-domain represent ations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)
* SelfAttention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* HAN: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
* CapsuleNet: [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* Transformer(encode or decode): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Bert:                  [BERT: Pre-trainingofDeepBidirectionalTransformersfor LanguageUnderstanding]()
* Xlnet:                 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
* Albert:                [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)

# 参考/感谢
* 文本分类项目:   [https://github.com/mosu027/TextClassification](https://github.com/mosu027/TextClassification)
* 文本分类看山杯: [https://github.com/brightmart/text_classification](https://github.com/brightmart/text_classification)
* Kashgari项目: [https://github.com/BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* 文本分类Ipty : [https://github.com/lpty/classifier](https://github.com/lpty/classifier)
* keras文本分类: [https://github.com/ShawnyXiao/TextClassification-Keras](https://github.com/ShawnyXiao/TextClassification-Keras)
* keras文本分类: [https://github.com/AlexYangLi/TextClassification](https://github.com/AlexYangLi/TextClassification)
* CapsuleNet模型: [https://github.com/bojone/Capsule](https://github.com/bojone/Capsule)
* transformer模型: [https://github.com/CyberZHG/keras-transformer](https://github.com/CyberZHG/keras-transformer)
* keras_albert_model: [https://github.com/TinkerMob/keras_albert_model](https://github.com/TinkerMob/keras_albert_model)


# 训练简单调用:
```python
from keras_textclassification import train
train(graph='TextCNN', # 必填, 算法名, 可选"ALBERT","BERT","XLNET","FASTTEXT","TEXTCNN","CHARCNN",
                       # "TEXTRNN","RCNN","DCNN","DPCNN","VDCNN","CRNN","DEEPMOJI",
                       # "SELFATTENTION", "HAN","CAPSULE","TRANSFORMER"
     label=17,         # 必填, 类别数, 训练集和测试集合必须一样
     path_train_data=None, # 必填, 训练数据文件, csv格式, 必须含'label,ques'头文件, 详见keras_textclassification/data
     path_dev_data=None, # 必填, 测试数据文件, csv格式, 必须含'label,ques'头文件, 详见keras_textclassification/data
     rate=1,             # 可填, 训练数据选取比例
     hyper_parameters=None) # 可填, json格式, 超参数, 默认embedding为'char','random'
```
        
# Train&Usage(调用)

```python
# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from keras_textclassification.conf.path_config import path_model_dir
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText, delete_file
# 模型图
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph
# 计算时间
import time


# 可配置地址
# path_model_dir = 'Y:/tet_keras_textclassification/'
path_model = path_model_dir + '/textcnn.model'
path_fineture = path_model_dir + '/fineture.embedding'
path_hyper_parameters = path_model_dir + '/hyper_parameters.json'

# 输入训练验证文件地址,sample数据集label填17
# path_train = path_model_dir + 'data/train.csv'
# path_valid = path_model_dir + 'data/val.csv'
# # or 输入训练/预测list, 这时候label选择填3
path_train = ['游戏,斩 魔仙 者 称号 怎么 得来 的', '文化,我爱你 古文 怎么 说', '健康,牙龈 包住 牙齿 怎么办']
path_valid = ['娱乐,李克勤 什么 歌 好听', '电脑,UPS 电源 工作 原理', '文化,我爱你 古文 怎么 说 的 呢']
# 会删除存在的model目录下的所有文件
# path_model_dir = 'Y:/tet_keras_textclassification/model/'


def train(hyper_parameters=None, rate=1.0):
    if not hyper_parameters:
        # 可配置参数
        hyper_parameters = {
        'len_max': 50,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 本地win10-4G设为20就好, 过大小心OOM
        'embed_size': 300,  # 字/词向量维度, bert取768, word取300, char可以更小些
        'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
        'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
        'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
        'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
        'gpu_memory_fraction': 0.66, #gpu使用率
        'model': {'label': 3,  # 类别数
                  'batch_size': 5,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                  'dropout': 0.5,  # 随机失活, 概率
                  'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
                  'decay_rate': 0.9,  # 学习率衰减系数, 乘法
                  'epochs': 20,  # 训练最大轮次
                  'patience': 3, # 早停,2-3就好
                  'lr': 5e-5,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                  'l2': 1e-9,  # l2正则化
                  'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数
                  'loss': 'categorical_crossentropy',  # 损失函数
                  'metrics': 'accuracy',  # 保存更好模型的评价标准
                  'is_training': True,  # 训练后者是测试模型
                  'model_path': path_model,
                  # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                  'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                  'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                  },
        'embedding': {'layer_indexes': [12], # bert取的层数
                      # 'corpus_path': '', # embedding预训练数据地址,不配则会默认取conf里边默认的地址
                        },
        'data':{'train_data': path_train, # 训练数据
                'val_data': path_valid    # 验证数据
                },
    }

    # 删除先前存在的模型和embedding微调模型等
    delete_file(path_model_dir)
    time_start = time.time()
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # 数据预处理
    pt = PreprocessText()
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
    print("耗时:" + str(time.time()-time_start))


if __name__=="__main__":
    train(rate=1)
    # 注意: 4G的1050Ti的GPU、win10下batch_size=32,len_max=20, gpu<=0.87, 应该就可以bert-fineture了。
    # 全量数据训练一轮(batch_size=32),就能达到80%准确率(验证集), 效果还是不错的
    # win10下出现过错误,gpu、len_max、batch_size配小一点就好:ailed to allocate 3.56G (3822520832 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
```


# Predict&Usage(调用)

```python
# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText, read_and_process, load_json
# 模型图
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph

from keras_textclassification.conf.path_config import path_model_dir
# 计算时间
import time

import numpy as np


def pred_input(path_hyper_parameter):
    # 输入预测
    # 加载超参数
    hyper_parameters = load_json(path_hyper_parameter)
    pt = PreprocessText()
    # 模式初始化和加载
    graph = Graph(hyper_parameters)
    graph.load_model()
    ra_ed = graph.word_embedding
    ques = '我要打王者荣耀'
    # str to token
    ques_embed = ra_ed.sentence2idx(ques)
    if hyper_parameters['embedding_type'] == 'bert':
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
        if hyper_parameters['embedding_type'] == 'bert':
            x_val_1 = np.array([ques_embed[0]])
            x_val_2 = np.array([ques_embed[1]])
            x_val = [x_val_1, x_val_2]
        else:
            x_val = ques_embed
        pred = graph.predict(x_val)
        pre = pt.prereocess_idx(pred[0])
        print(pre)


if __name__=="__main__":
    # 可输入 input 预测
    pred_input(path_hyper_parameter=path_model_dir + '/hyper_parameters.json')
```


*希望对你有所帮助!
