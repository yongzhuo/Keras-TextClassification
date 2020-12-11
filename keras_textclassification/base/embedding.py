# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 11:29
# @author   :Mo
# @function :embeddings of model, base embedding of random, word2vec or bert


from keras_textclassification.conf.path_config import path_embedding_vector_word2vec_char, path_embedding_vector_word2vec_word
from keras_textclassification.conf.path_config import path_embedding_bert, path_embedding_xlnet, path_embedding_albert
from keras_textclassification.conf.path_config import path_embedding_random_char, path_embedding_random_word
from keras_textclassification.data_preprocess.text_preprocess import extract_chinese
from keras_textclassification.data_preprocess.text_preprocess import get_ngram

from keras_textclassification.keras_layers.non_mask_layer import NonMaskingLayer
from keras.layers import Add, Embedding, Lambda
from gensim.models import KeyedVectors
from keras.models import Input, Model
import keras.backend as K

import numpy as np
import codecs
import jieba
import json
import os


class BaseEmbedding:
    def __init__(self, hyper_parameters):
        self.len_max = hyper_parameters.get('len_max', 50)  # 文本最大长度, 建议25-50
        self.embed_size = hyper_parameters.get('embed_size', 300)  # 嵌入层尺寸
        self.vocab_size = hyper_parameters.get('vocab_size', 30000)  # 字典大小, 这里随便填的，会根据代码里修改
        self.trainable = hyper_parameters.get('trainable', False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.level_type = hyper_parameters.get('level_type', 'char')  # 还可以填'word'
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')  # 词嵌入方式，可以选择'xlnet'、'bert'、'random'、'word2vec'

        # 自适应, 根据level_type和embedding_type判断corpus_path
        if self.level_type == "word":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_word)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_word)
            elif self.embedding_type == "bert":
                raise RuntimeError("bert level_type is 'char', not 'word'")
            elif self.embedding_type == "xlnet":
                raise RuntimeError("xlnet level_type is 'char', not 'word'")
            elif self.embedding_type == "albert":
                raise RuntimeError("albert level_type is 'char', not 'word'")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "char":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_char)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_vector_word2vec_char)
            elif self.embedding_type == "bert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_bert)
            elif self.embedding_type == "xlnet":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_xlnet)
            elif self.embedding_type == "albert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_albert)
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "ngram":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path')
                if not self.corpus_path:
                    raise RuntimeError("corpus_path must exists!")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        else:
            raise RuntimeError("level_type must be 'char' or 'word'")
        # 定义的符号
        self.ot_dict = {'[PAD]': 0,
                        '[UNK]': 1,
                        '[BOS]': 2,
                        '[EOS]': 3, }
        self.deal_corpus()
        self.build()

    def deal_corpus(self):  # 处理语料
        pass

    def build(self):
        self.token2idx = {}
        self.idx2token = {}

    def sentence2idx(self, text, second_text=None):
        if second_text:
            second_text = "[SEP]" + str(second_text).upper()
        # text = extract_chinese(str(text).upper())
        text = str(text).upper()

        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index

    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)


class RandomEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.ngram_ns = hyper_parameters['embedding'].get('ngram_ns', [1, 2, 3]) # ngram信息, 根据预料获取
        # self.path = hyper_parameters.get('corpus_path', path_embedding_random_char)
        super().__init__(hyper_parameters)

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

        elif os.path.exists(self.corpus_path):
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                terms = fd.readlines()
                for term_one in terms:
                    if self.level_type == 'char':
                        text = list(term_one.replace(' ', '').strip())
                    elif self.level_type == 'word':
                        text = list(jieba.cut(term_one, cut_all=False, HMM=False))
                    elif self.level_type == 'ngram':
                        text = get_ngram(term_one, ns=self.ngram_ns)
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word', 'char', 'ngram'")
                    for text_one in text:
                        if text_one not in token2idx:
                            count = count + 1
                            token2idx[text_one] = count
        else:
            raise RuntimeError("your input corpus_path is wrong, it must be 'dict' or 'corpus'")
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        self.input = Input(shape=(self.len_max,), dtype='int32')
        self.output = Embedding(self.vocab_size+1,
                                self.embed_size,
                                input_length=self.len_max,
                                trainable=self.trainable,
                                )(self.input)
        self.model = Model(self.input, self.output)

    def sentence2idx(self, text, second_text=""):
        if second_text:
            second_text = "[SEP]" + str(second_text).upper()
        # text = extract_chinese(str(text).upper()+second_text)
        text =str(text).upper() + second_text
        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = list(jieba.cut(text, cut_all=False, HMM=False))
        elif self.level_type == 'ngram':
            text = get_ngram(text, ns=self.ngram_ns)
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        # text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index


class WordEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        # self.path = hyper_parameters.get('corpus_path', path_embedding_vector_word2vec)
        super().__init__(hyper_parameters)

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

        self.vocab_size = len(self.token2idx)
        embedding_matrix = np.array(embedding_matrix)
        self.input = Input(shape=(self.len_max,), dtype='int32')

        self.output = Embedding(self.vocab_size,
                                self.embed_size,
                                input_length=self.len_max,
                                weights=[embedding_matrix],
                                trainable=self.trainable)(self.input)
        self.model = Model(self.input, self.output)


class BertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        import keras_bert

        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        check_point_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        print('load bert model start!')
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              check_point_path,
                                                              seq_len=self.len_max,
                                                              trainable=self.trainable)
        print('load bert model end!')
        # bert model all layers
        layer_dict = [6]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        print(layer_dict)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(13)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0] - 1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(13)]
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
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def build_keras4bert(self):
        import bert4keras
        from bert4keras.models import build_transformer_model
        from bert4keras.tokenizers import Tokenizer,load_vocab
        import os
        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        checkpoint_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        self.model = bert4keras.models.build_transformer_model(config_path=config_path,
                                                               checkpoint_path=checkpoint_path)

        # 加载并精简词表，建立分词器
        self.token_dict, keep_tokens = load_vocab(
            dict_path=dict_path,
            simplified=True,
            startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        )
        self.vocab_size = len(self.token_dict)
        self.tokenizer = Tokenizer(self.token_dict, do_lower_case=True)

    def sentence2idx(self, text, second_text=None):
        text = extract_chinese(str(text).upper())
        text = str(text).upper()
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        return [input_id, input_type_id]

        # input_id, input_type_id = self.tokenizer.encode(first_text=text,
        #                                                 second_text=second_text,
        #                                                 max_length=self.len_max,
        #                                                 first_length=self.len_max)
        #
        # input_mask = [0 if ids == 0 else 1 for ids in input_id]
        # return [input_id, input_type_id, input_mask]


class XlnetEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [24])
        self.xlnet_embed = hyper_parameters['embedding'].get('xlnet_embed', {})
        self.batch_size = hyper_parameters['model'].get('batch_size', 2)
        super().__init__(hyper_parameters)

    def build_config(self, path_config: str=None):
        # reader config of bert
        self.configs = {}
        if path_config is not None:
            self.configs.update(json.load(open(path_config)))

    def build(self):
        from keras_xlnet import load_trained_model_from_checkpoint, set_custom_objects
        from keras_xlnet import Tokenizer, ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI

        self.embedding_type = 'xlnet'
        self.checkpoint_path = os.path.join(self.corpus_path, 'xlnet_model.ckpt')
        self.config_path = os.path.join(self.corpus_path, 'xlnet_config.json')
        self.spiece_model = os.path.join(self.corpus_path, 'spiece.model')

        self.attention_type = self.xlnet_embed.get('attention_type', 'bi')  # or 'uni'
        self.attention_type = ATTENTION_TYPE_BI if self.attention_type == 'bi' else ATTENTION_TYPE_UNI
        self.memory_len = self.xlnet_embed.get('memory_len', 0)
        self.target_len = self.xlnet_embed.get('target_len', 5)
        print('load xlnet model start!')
        # 模型加载
        model = load_trained_model_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                   attention_type=self.attention_type,
                                                   in_train_phase=self.trainable,
                                                   config_path=self.config_path,
                                                   memory_len=self.memory_len,
                                                   target_len=self.target_len,
                                                   batch_size=self.batch_size,
                                                   mask_index=0)
        #
        set_custom_objects()
        self.build_config(self.config_path)
        # 字典加载
        self.tokenizer = Tokenizer(self.spiece_model)
        # # debug时候查看layers
        # self.model_layers = model.layers
        # len_layers = self.model_layers.__len__()
        # print(len_layers)
        num_hidden_layers = self.configs.get("n_layer", 12)

        layer_real = [i for i in range(num_hidden_layers)] + [-i for i in range(num_hidden_layers)]
        # 简要判别一下
        self.layer_indexes = [i if i in layer_real else -2 for i in self.layer_indexes]
        output_layer = "FeedForward-Normal-{0}"
        layer_dict = [model.get_layer(output_layer.format(i + 1)).get_output_at(node_index=0)
                          for i in range(num_hidden_layers)]

        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，取得不正确的话就取倒数第二层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in layer_real:
                encoder_layer = layer_dict[self.layer_indexes[0]]
            else:
                encoder_layer = layer_dict[-1]
        # 否则遍历需要取的层，把所有层的weight取出来并加起来shape:768*层数
        else:
            # layer_indexes must be [0, 1, 2,3,......24]
            all_layers = [layer_dict[lay] if lay in layer_real
                          else layer_dict[-1] # 如果给出不正确，就默认输出倒数第一层
                          for lay in self.layer_indexes]
            print(self.layer_indexes)
            print(all_layers)
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)

            # def xlnet_concat(x):
            #     x_concat = K.concatenate(x, axis=1)
            #     return x_concat
            # encoder_layer = Lambda(xlnet_concat, name='xlnet_concat')(all_layers)

        self.output = NonMaskingLayer()(encoder_layer)
        self.input = model.inputs
        self.model = Model(self.input, self.output)
        print("load KerasXlnetEmbedding end")
        model.summary(132)

        self.embedding_size = self.model.output_shape[-1]
        self.vocab_size = len(self.tokenizer.sp)

    def sentence2idx(self, text, second_text=None):
        # text = extract_chinese(str(text).upper())
        text = str(text).upper()
        tokens = self.tokenizer.encode(text)
        tokens = tokens + [0] * (self.target_len - len(tokens)) \
                               if len(tokens) < self.target_len \
                               else tokens[0:self.target_len]
        token_input = np.expand_dims(np.array(tokens), axis=0)
        segment_input = np.zeros_like(token_input)
        memory_length_input = np.zeros((1, 1)) # np.array([[self.memory_len]]) # np.zeros((1, 1))
        masks = [1] * len(tokens) + ([0] * (self.target_len - len(tokens))
                                                   if len(tokens) < self.target_len else [])
        mask_input = np.expand_dims(np.array(masks), axis=0)
        if self.trainable:
            return [token_input, segment_input, memory_length_input, mask_input]
        else:
            return [token_input, segment_input, memory_length_input]


class AlbertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        from keras_textclassification.keras_layers.albert.albert import load_brightmart_albert_zh_checkpoint
        import keras_bert

        self.embedding_type = 'albert'
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        print('load bert model start!')
        # 简要判别一下
        # self.layer_indexes = [i if i in layer_real else -2 for i in self.layer_indexes]
        self.model = load_brightmart_albert_zh_checkpoint(self.corpus_path,
                                                     training=self.trainable,
                                                     seq_len=self.len_max,
                                                     output_layers = None) # self.layer_indexes)
        import json
        config = {}
        for file_name in os.listdir(self.corpus_path):
            if file_name.startswith('bert_config.json'):
                with open(os.path.join(self.corpus_path, file_name)) as reader:
                    config = json.load(reader)
                break

        num_hidden_layers = config.get("num_hidden_layers", 0)
        layer_real  = [i for i in range(num_hidden_layers)] + [-i for i in range(num_hidden_layers)]
        self.layer_indexes = [i if i in layer_real else -2 for i in self.layer_indexes]

        # self.input = self.model.inputs
        # self.output = self.model.outputs[0]
        model_l = self.model.layers
        print('load bert model end!')
        # albert model all layers
        layer_dict = [4, 8, 11, 13]
        layer_0 = 13
        for i in range(num_hidden_layers):
            layer_0 = layer_0 + 1
            layer_dict.append(layer_0)
        # layer_dict.append(34)
        print(layer_dict)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = self.model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in layer_real:
                encoder_layer = self.model.get_layer(index=layer_dict[self.layer_indexes[0]]).output
            else:
                encoder_layer = self.model.get_layer(index=layer_dict[-2]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [self.model.get_layer(index=layer_dict[lay]).output if lay in layer_real
                          else self.model.get_layer(index=layer_dict[-2]).output  # 如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = self.model.inputs
        self.model = Model(self.input, self.output)

        # self.embedding_size = self.model.output_shape[-1]

        # reader tokenizer
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text, second_text=""):
        # text = extract_chinese(str(text).upper())
        text = str(text).upper()
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        # input_mask = [0 if ids == 0 else 1 for ids in input_id]
        # return input_id, input_type_id, input_mask
        return [input_id, input_type_id]

