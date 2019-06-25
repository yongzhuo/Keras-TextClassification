# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 11:29
# @author   :Mo
# @function :embeddings of model, base embedding of random word2vec or bert


import codecs
import os

import jieba
import keras_bert
from gensim.models import KeyedVectors
from keras.engine import Layer
from keras.layers import Concatenate
from keras.layers import Embedding
from keras.layers import Add
from keras.models import Input
from keras.models import Model

import numpy as np


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class BaseEmbedding:
    def __init__(self, hyper_parameters):
        self.corpus_path = hyper_parameters['embedding'].get('corpus_path', 'corpus_path') # 'dict' or 'corpus'
        self.level_type = hyper_parameters['embedding'].get('level_type', 'char') # 还可以填'word'
        self.vocab_size = hyper_parameters['embedding'].get('vocab_size', 30000) #
        self.embed_size = hyper_parameters['embedding'].get('embed_size', 300) # word 150万所以300；中文char 2000所以30左右
        self.len_max = hyper_parameters['embedding'].get('len_max', 50) # 建议25-50
        self.ot_dict = { 'PAD': 0,
                         'UNK': 1,
                         'BOS': 2,
                         'EOS': 3, }
        self.deal_corpus()
        self.build()

    def deal_corpus(self):
        pass

    def build(self):
        self.token2idx = {}
        self.idx2token = {}

    def sentence2idx(self, text):
        text = str(text)
        if self.level_type == 'char':
            text = list(text.replace(' ', '').strip())
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['UNK'] for
                          text_char in text] + [self.token2idx['PAD'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['UNK'] for
                          text_char in text[0:self.len_max]]
        return text_index

    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['UNK'] for id in idx]
        return "".join(text_idx)


class RandomEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        # self.path = hyper_parameters.get('corpus_path', path_embedding_random_char)

    def deal_corpus(self):
        token2idx = self.ot_dict.copy()
        count = 3
        if 'term' in self.corpus_path:
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                while True:
                    term_one = fd.readline()
                    if not term_one:
                        break
                    term_one = term_one.strip()
                    if term_one not in token2idx:
                        count = count + 1
                        token2idx[term_one] = count

        elif 'corpus' in self.corpus_path:
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                terms = fd.readlines()
                for term_one in terms:
                    if self.level_type == 'char':
                        text = list(term_one.replace(' ', '').strip())
                    elif self.level_type == 'word':
                        text = list(jieba.cut(term_one, cut_all=False, HMM=True))
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
                    for text_one in text:
                        if term_one not in token2idx:
                            count = count + 1
                            token2idx[text_one] = count
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'dict' or 'corpus'")
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        self.input = Input(shape=(self.len_max, ), dtype='int32')
        self.output = Embedding(self.vocab_size,
                            self.embed_size,
                            input_length=self.len_max,
                            trainable=True)(self.input)
        self.model = Model(self.input, self.output)


class WordEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        # self.path = hyper_parameters.get('corpus_path', path_embedding_vector_word2vec)

    def build(self, **kwargs):
        self.embedding_type = 'word2vec'
        print("load word2vec start!")
        self.key_vector = KeyedVectors.load_word2vec_format(self.corpus_path, **kwargs)
        print("load word2vec end!")
        self.embed_size = self.key_vector.vector_size

        self.token2idx = self.ot_dict.copy()
        embedding_matrix = []
        # 首先加self.token2idx中的四个[PAD]、[UNK]、[BOS]、[EOS]
        embedding_matrix.append(np.zeros(self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))

        for word in self.key_vector.index2entity:
            self.token2idx[word] = len(self.token2idx)
            embedding_matrix.append(self.key_vector[word])

        # self.token2idx = self.token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

        len_token2idx = len(self.token2idx)
        embedding_matrix = np.array(embedding_matrix)
        self.input = Input(shape=(self.len_max,), dtype='int32')

        self.output = Embedding(len_token2idx,
                            self.embed_size,
                            input_length=self.len_max,
                            weights=[embedding_matrix],
                            trainable=False)(self.input)
        self.model = Model(self.input, self.output)


class BertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        check_point_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        print('load bert model start!')
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              check_point_path,
                                                              seq_len=self.len_max)
        print('load bert model end!')
        # bert model all layers
        layer_dict = [7]
        layer_0 = 7
        for i in range(11):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(12)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0] - 1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(12)]
                          else model.get_layer(index=layer_dict[-1]).output  # 如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = model.inputs
        self.model = Model(self.input, self.output)

        self.embedding_size = self.model.output_shape[-1]
        # word2idx = {}
        # with open(dict_path, 'r', encoding='utf-8') as f:
        #     words = f.read().splitlines()
        # for idx, word in enumerate(words):
        #     word2idx[word] = idx
        # for key, value in self.ot_dict.items():
        #     word2idx[key] = value
        #
        # self.token2idx = word2idx

        # reader tokenizer
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text):
        input_id, input_type_id = self.tokenizer.encode(first=text, max_len=self.len_max)
        # input_mask = [0 if ids == 0 else 1 for ids in input_id]
        # return input_id, input_type_id, input_mask
        return [input_id, input_type_id]
