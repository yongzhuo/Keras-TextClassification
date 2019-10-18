# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/22 7:35
# @author   :TinkerMob
# @function :keras_albert_model
# @code     :code from https://github.com/TinkerMob/keras_albert_model


from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax
from keras_bert import get_custom_objects as get_bert_custom_objects
from keras_position_wise_feed_forward import FeedForward
from keras_layer_normalization import LayerNormalization
from keras_bert.activations.gelu_fallback import gelu
from keras_multi_head import MultiHeadAttention
from keras_bert.layers import Masked, Extract
from keras_pos_embd import PositionEmbedding
from keras_bert.backend import keras
import tensorflow as tf
import numpy as np
import json
import os


__all__ = [
    'get_custom_objects', 'build_albert',
    'load_brightmart_albert_zh_checkpoint',
]


def get_custom_objects():
    custom_objects = get_bert_custom_objects()
    custom_objects['AdaptiveEmbedding'] = AdaptiveEmbedding
    custom_objects['AdaptiveSoftmax'] = AdaptiveSoftmax
    return custom_objects


def build_albert(token_num,
                 pos_num=512,
                 seq_len=512,
                 embed_dim=128,
                 hidden_dim=768,
                 transformer_num=12,
                 head_num=12,
                 feed_forward_dim=3072,
                 dropout_rate=0.1,
                 attention_activation=None,
                 feed_forward_activation='gelu',
                 training=True,
                 trainable=None,
                 output_layers=None):
    """Get ALBERT model.
    See: https://arxiv.org/pdf/1909.11942.pdf
    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param seq_len: Maximum length of the input sequence or None.
    :param embed_dim: Dimensions of embeddings.
    :param hidden_dim: Dimensions of hidden layers.
    :param transformer_num: Number of transformers.
    :param head_num: Number of heads in multi-head attention
                    in each transformer.
    :param feed_forward_dim: Dimension of the feed forward layer
                             in each transformer.
    :param dropout_rate: Dropout rate.
    :param attention_activation: Activation for attention layers.
    :param feed_forward_activation: Activation for feed-forward layers.
    :param training: A built model with MLM and NSP outputs will be returned
                     if it is `True`, otherwise the input layers and the last
                     feature extraction layer will be returned.
    :param trainable: Whether the model is trainable.
    :param output_layers: A list of indices of output layers.
    """
    if attention_activation == 'gelu':
        attention_activation = gelu
    if feed_forward_activation == 'gelu':
        feed_forward_activation = gelu
    if trainable is None:
        trainable = training

    def _trainable(_layer):
        if isinstance(trainable, (list, tuple, set)):
            for prefix in trainable:
                if _layer.name.startswith(prefix):
                    return True
            return False
        return trainable

    # Build inputs
    input_token = keras.layers.Input(shape=(seq_len,), name='Input-Token')
    input_segment = keras.layers.Input(shape=(seq_len,), name='Input-Segment')
    inputs = [input_token, input_segment]

    # Build embeddings
    embed_token, embed_weights, embed_projection = AdaptiveEmbedding(
        input_dim=token_num,
        output_dim=hidden_dim,
        embed_dim=embed_dim,
        mask_zero=True,
        trainable=trainable,
        return_embeddings=True,
        return_projections=True,
        name='Embed-Token',
    )(input_token)
    embed_segment = keras.layers.Embedding(
        input_dim=2,
        output_dim=hidden_dim,
        trainable=trainable,
        name='Embed-Segment',
    )(input_segment)
    embed_layer = keras.layers.Add(name='Embed-Token-Segment')(
        [embed_token, embed_segment])
    embed_layer = PositionEmbedding(
        input_dim=pos_num,
        output_dim=hidden_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='Embedding-Position',
    )(embed_layer)

    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    embed_layer = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm',
    )(dropout_layer)

    # Build shared transformer
    attention_layer = MultiHeadAttention(
        head_num=head_num,
        activation=attention_activation,
        name='Attention',
    )
    attention_normal = LayerNormalization(name='Attention-Normal')
    feed_forward_layer = FeedForward(
        units=feed_forward_dim,
        activation=feed_forward_activation,
        name='Feed-Forward'
    )
    feed_forward_normal = LayerNormalization(name='Feed-Forward-Normal')

    transformed = embed_layer
    transformed_layers = []
    for i in range(transformer_num):
        attention_input = transformed
        transformed = attention_layer(transformed)
        if dropout_rate > 0.0:
            transformed = keras.layers.Dropout(
                rate=dropout_rate,
                name='Attention-Dropout-{}'.format(i + 1),
            )(transformed)
        transformed = keras.layers.Add(
            name='Attention-Add-{}'.format(i + 1),
        )([attention_input, transformed])
        transformed = attention_normal(transformed)

        feed_forward_input = transformed
        transformed = feed_forward_layer(transformed)
        if dropout_rate > 0.0:
            transformed = keras.layers.Dropout(
                rate=dropout_rate,
                name='Feed-Forward-Dropout-{}'.format(i + 1),
            )(transformed)
        transformed = keras.layers.Add(
            name='Feed-Forward-Add-{}'.format(i + 1),
        )([feed_forward_input, transformed])
        transformed = feed_forward_normal(transformed)
        transformed_layers.append(transformed)

    if training:
        # Build tasks
        mlm_dense_layer = keras.layers.Dense(
            units=hidden_dim,
            activation=feed_forward_activation,
            name='MLM-Dense',
        )(transformed)
        mlm_norm_layer = LayerNormalization(name='MLM-Norm')(mlm_dense_layer)
        mlm_pred_layer = AdaptiveSoftmax(
            input_dim=hidden_dim,
            output_dim=token_num,
            embed_dim=embed_dim,
            bind_embeddings=True,
            bind_projections=True,
            name='MLM-Sim',
        )([mlm_norm_layer, embed_weights, embed_projection])
        masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
        extract_layer = Extract(index=0, name='Extract')(transformed)
        nsp_dense_layer = keras.layers.Dense(
            units=hidden_dim,
            activation='tanh',
            name='SOP-Dense',
        )(extract_layer)
        nsp_pred_layer = keras.layers.Dense(
            units=2,
            activation='softmax',
            name='SOP',
        )(nsp_dense_layer)
        model = keras.models.Model(
            inputs=inputs,
            outputs=[masked_layer, nsp_pred_layer])
        for layer in model.layers:
            layer.trainable = _trainable(layer)
        return model
    if output_layers is not None:
        if isinstance(output_layers, list):
            output_layers = [
                transformed_layers[index] for index in output_layers]
            output = keras.layers.Concatenate(
                name='Output',
            )(output_layers)
        else:
            output = transformed_layers[output_layers]
        model = keras.models.Model(inputs=inputs, outputs=output)
        return model
    model = keras.models.Model(inputs=inputs, outputs=transformed)
    for layer in model.layers:
        layer.trainable = _trainable(layer)
    return inputs, transformed


def load_brightmart_albert_zh_checkpoint(checkpoint_path, **kwargs):
    """Load checkpoint from https://github.com/brightmart/albert_zh
    :param checkpoint_path: path to checkpoint folder.
    :param kwargs: arguments for albert model.
    :return:
    """
    config = {}
    for file_name in os.listdir(checkpoint_path):
        if file_name.startswith('bert_config.json'):
            with open(os.path.join(checkpoint_path, file_name)) as reader:
                config = json.load(reader)
            break

    def _set_if_not_existed(key, value):
        if key not in kwargs:
            kwargs[key] = value

    # 修改部分,必须输入is_training, len_max
    training = kwargs['training']
    # config['max_position_embeddings'] = config['max_position_embeddings'] = kwargs['len_max']
    _set_if_not_existed('training', True)
    _set_if_not_existed('token_num', config['vocab_size'])
    _set_if_not_existed('pos_num', config['max_position_embeddings'])
    _set_if_not_existed('seq_len', config['max_position_embeddings'])
    _set_if_not_existed('embed_dim', config['embedding_size'])
    _set_if_not_existed('hidden_dim', config['hidden_size'])
    _set_if_not_existed('transformer_num', config['num_hidden_layers'])
    _set_if_not_existed('head_num', config['num_attention_heads'])
    _set_if_not_existed('feed_forward_dim', config['intermediate_size'])
    _set_if_not_existed('dropout_rate', config['hidden_dropout_prob'])
    _set_if_not_existed('feed_forward_activation', config['hidden_act'])

    model = build_albert(**kwargs)
    if not training:
        inputs, outputs = model
        model = keras.models.Model(inputs, outputs)

    def _checkpoint_loader(checkpoint_file):
        def _loader(name):
            return tf.train.load_variable(checkpoint_file, name)
        return _loader

    loader = _checkpoint_loader(
        os.path.join(checkpoint_path, 'bert_model.ckpt'))

    model.get_layer(name='Embed-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/word_embeddings_2'),
    ])
    model.get_layer(name='Embed-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])

    model.get_layer(name='Attention').set_weights([
        loader('bert/encoder/layer_shared/attention/self/query/kernel'),
        loader('bert/encoder/layer_shared/attention/self/query/bias'),
        loader('bert/encoder/layer_shared/attention/self/key/kernel'),
        loader('bert/encoder/layer_shared/attention/self/key/bias'),
        loader('bert/encoder/layer_shared/attention/self/value/kernel'),
        loader('bert/encoder/layer_shared/attention/self/value/bias'),
        loader('bert/encoder/layer_shared/attention/output/dense/kernel'),
        loader('bert/encoder/layer_shared/attention/output/dense/bias'),
    ])
    model.get_layer(name='Attention-Normal').set_weights([
        loader('bert/encoder/layer_shared/attention/output/LayerNorm/gamma'),
        loader('bert/encoder/layer_shared/attention/output/LayerNorm/beta'),
    ])
    model.get_layer(name='Feed-Forward').set_weights([
        loader('bert/encoder/layer_shared/intermediate/dense/kernel'),
        loader('bert/encoder/layer_shared/intermediate/dense/bias'),
        loader('bert/encoder/layer_shared/output/dense/kernel'),
        loader('bert/encoder/layer_shared/output/dense/bias'),
    ])
    model.get_layer(name='Feed-Forward-Normal').set_weights([
        loader('bert/encoder/layer_shared/output/LayerNorm/gamma'),
        loader('bert/encoder/layer_shared/output/LayerNorm/beta'),
    ])

    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])

        model.get_layer(name='SOP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='SOP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])

    return model