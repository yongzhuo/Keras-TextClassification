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
    - Electra-fineture(todo)
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
    - SWEM
    - LEAM
    - TextGCN(todo)


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
       - chinese_xlnet_base_L-12_H-768_A-12/(哈工大预训练的中文xlnet模型[https://github.com/ymcui/Chinese-PreTrained-XLNet],12层)
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
* RoBERTa:               [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* ELECTRA:               [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
* TextGCN:               [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)


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

# Reference
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@misc{Keras-TextClassification,
    howpublished = {\url{https://github.com/yongzhuo/Keras-TextClassification}},
    title = {Keras-TextClassification},
    author = {Yongzhuo Mo},
    publisher = {GitHub},
    year = {2019}
}
```

*希望对你有所帮助!
