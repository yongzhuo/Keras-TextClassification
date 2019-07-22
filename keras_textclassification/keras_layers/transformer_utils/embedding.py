# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/7/22 10:00
# @author   :Mo
# @function :

from keras.layers import Embedding, Layer
import keras.backend as K
import keras


class EmbeddingRet(Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [super(EmbeddingRet, self).compute_output_shape(input_shape),
                (self.input_dim, self.output_dim),
                ]

    def compute_mask(self, inputs, mask=None):
        return [super(EmbeddingRet, self).compute_mask(inputs, mask),
                None,
                ]

    def call(self, inputs):
        return [super(EmbeddingRet, self).call(inputs),
                self.embeddings,
                ]


class EmbeddingSim(Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 use_bias=True,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
        :param use_bias: Whether to use bias term.
        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSim, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self):
        config = {'use_bias': self.use_bias,
                  'initializer': keras.initializers.serialize(self.initializer),
                  'regularizer': keras.regularizers.serialize(self.regularizer),
                  'constraint': keras.constraints.serialize(self.constraint),
                  }
        base_config = super(EmbeddingSim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.use_bias:
            embed_shape = input_shape[1]
            token_num = embed_shape[0]
            self.bias = self.add_weight(shape=(token_num,),
                                        initializer=self.initializer,
                                        regularizer=self.regularizer,
                                        constraint=self.constraint,
                                        name='bias',
                                        )
        super(EmbeddingSim, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_shape, embed_shape = input_shape
        token_num = embed_shape[0]
        return feature_shape[:-1] + (token_num,)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        outputs = K.dot(inputs, K.transpose(embeddings))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        return keras.activations.softmax(outputs)
