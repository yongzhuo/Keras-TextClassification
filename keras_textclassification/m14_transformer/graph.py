# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of CapsuleNet


from keras.layers import Dropout, Dense, Flatten
from keras.models import Model
import keras

from keras_textclassification.keras_layers.transformer_utils.triangle_position_embedding import TriglePositiomEmbedding
from keras_textclassification.keras_layers.transformer_utils.embedding import EmbeddingRet
from keras_textclassification.keras_layers.transformer import build_encoders

from keras_textclassification.keras_layers.non_mask_layer import NonMaskingLayer

from keras_textclassification.base.graph import graph


class TransformerEncodeGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.encoder_num = hyper_parameters["model"].get('encoder_num', 2)
        self.head_num = hyper_parameters["model"].get('head_num', 6)
        self.hidden_dim = hyper_parameters["model"].get('hidden_dim', 3072)
        self.attention_activation = hyper_parameters["model"].get('attention_activation', 'relu')
        self.feed_forward_activation = hyper_parameters["model"].get('feed_forward_activation', 'relu')
        self.use_adapter = hyper_parameters["model"].get('use_adapter', False)
        self.adapter_units = hyper_parameters["model"].get('adapter_units', 768)
        self.adapter_activation  = hyper_parameters["model"].get('adapter_activation', 'relu')
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters) # 这里的embedding不用,只提取token(即onehot-index)
        # embedding = self.word_embedding.output
        # embed_layer = SpatialDropout1D(self.dropout)(embedding)
        encoder_input = keras.layers.Input(shape=(self.len_max,), name='Encoder-Input')
        encoder_embed_layer = EmbeddingRet(input_dim=self.word_embedding.vocab_size,
                                           output_dim=self.word_embedding.embed_size,
                                           mask_zero=False,
                                           weights=None,
                                           trainable=self.trainable,
                                           name='Token-Embedding',)
        encoder_embedding = encoder_embed_layer(encoder_input)
        encoder_embed = TriglePositiomEmbedding(mode=TriglePositiomEmbedding.MODE_ADD,
                                                name='Encoder-Embedding',)(encoder_embedding[0])
        encoded_layer = build_encoders(encoder_num=self.encoder_num,
                                       input_layer=encoder_embed,
                                       head_num=self.head_num,
                                       hidden_dim=self.hidden_dim,
                                       attention_activation=self.activate_classify,
                                       feed_forward_activation=self.activate_classify,
                                       dropout_rate=self.dropout,
                                       trainable=self.trainable,
                                       use_adapter=self.use_adapter,
                                       adapter_units=self.adapter_units,
                                       adapter_activation=self.adapter_activation,
                                       )
        encoded_layer = NonMaskingLayer()(encoded_layer)
        encoded_layer_flat = Flatten()(encoded_layer)
        encoded_layer_drop = Dropout(self.dropout)(encoded_layer_flat)
        output = Dense(self.label, activation=self.activate_classify)(encoded_layer_drop)
        self.model = Model(inputs=encoder_input, outputs=output)
        self.model.summary(120)
