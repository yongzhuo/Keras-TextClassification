# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of DeepMoji(https://arxiv.org/abs/1708.00524)


from keras.layers import SpatialDropout1D, Activation, concatenate
from keras.layers import Dropout, Dense, Flatten
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional
from keras.engine import InputSpec, Layer
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras import regularizers

from keras_textclassification.base.graph import graph


class DeepMojiGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.num_rnn_layers = hyper_parameters['model'].get('num_rnn_layers', 2)
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 512)  # large, small is 300
        self.l2 = hyper_parameters['model'].get('l2', 0.0000032)
        self.dropout_spatial = hyper_parameters['model'].get('droupout_spatial', 0.2)
        self.activation_conv = hyper_parameters['model'].get('activation_conv', 'linear')
        self.return_attention = hyper_parameters['model'].get('return_attention', True)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络, a bit like RCNN, R
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        x = Activation('tanh')(x)

        # entire embedding channels are dropped out instead of the
        # normal Keras embedding dropout, which drops all channels for entire words
        # many of the datasets contain so few words that losing one or more words can alter the emotions completely
        x = SpatialDropout1D(self.dropout_spatial)(x)

        if self.rnn_units=="LSTM":
                layer_cell = LSTM
        elif self.rnn_units=="GRU":
                layer_cell = GRU
        elif self.rnn_units=="CuDNNLSTM":
                layer_cell = CuDNNLSTM
        elif self.rnn_units=="CuDNNGRU":
                layer_cell = CuDNNGRU
        else:
            layer_cell = GRU


        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output = Bidirectional(layer_cell(units=self.rnn_units,
                                                 return_sequences=True,
                                                 activation='relu',
                                                 kernel_regularizer=regularizers.l2(self.l2),
                                                 recurrent_regularizer=regularizers.l2(self.l2)
                                                 ), name="bi_lstm_0")(x)
        lstm_1_output = Bidirectional(layer_cell(units=self.rnn_units,
                                                 return_sequences=True,
                                                 activation='relu',
                                                 kernel_regularizer=regularizers.l2(self.l2),
                                                 recurrent_regularizer=regularizers.l2(self.l2)
                                                 ), name="bi_lstm_1")(lstm_0_output)
        x = concatenate([lstm_1_output, lstm_0_output, x])

        # if return_attention is True in AttentionWeightedAverage, an additional tensor
        # representing the weight at each timestep is returned
        weights = None
        x = AttentionWeightedAverage(name='attlayer', return_attention=self.return_attention)(x)
        if self.return_attention:
            x, weights = x

        x = Dropout(self.dropout)(x)
        x = Dense(128, activation="tanh")(x)
        x = Dropout(self.dropout)(x)

        # x = Flatten()(x)
        # 最后就是softmax
        dense_layer = Dense(self.label, activation=self.activate_classify)(x)
        output = [dense_layer]
        self.model = Model(self.word_embedding.input, output)
        self.model.summary(120)


class AttentionWeightedAverage(Layer):
    """
    codes from: https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
        }
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
