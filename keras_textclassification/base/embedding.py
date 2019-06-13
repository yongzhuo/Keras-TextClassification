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
from keras.models import Input
from keras.models import Model


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
        if 'term_char' in self.corpus_path:
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
        for word in self.key_vector.index2entity:
            self.token2idx[word] = len(self.token2idx)
            embedding_matrix.append(self.key_vector[word])
        self.token2idx = self.token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

        len_token2idx = len(self.token2idx)

        input_layer = Input(shape=(self.len_max,), dtype='int32')

        output = Embedding(len_token2idx,
                            self.embed_size,
                            input_length=self.len_max,
                            weights=[embedding_matrix],
                            trainable=False)(input_layer)
        self.model = Model(input_layer, output)


class BertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        # self.path = hyper_parameters.get('corpus_path', path_embedding_bert)

    def build(self):
        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        check_point_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              check_point_path,
                                                              seq_len=self.len_max)
        num_layers = len(model.layers)
        features_layers = [model.get_layer(index=num_layers-1+idx*8).output\
                            for idx in range(-3, 1)]
        embedding_layer = Concatenate(features_layers)
        output_layer = NonMaskingLayer()(embedding_layer)
        self.model = Model(model.inputs, output_layer)

        self.embedding_size = self.model.output_shape[-1]
        word2idx = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        for idx, word in enumerate(words):
            word2idx[word] = idx
        for key, value in self.ot_dict.items():
            word2idx[key] = word2idx[value]

        self.token2idx = word2idx

        # reader tokenizer
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text):
        input_id, input_type_id = self.tokenizer.encode(first=text, max_len=self.len_max)
        input_mask = [0 if ids == 0 else 1 for ids in input_id]
        return input_id, input_type_id, input_mask
