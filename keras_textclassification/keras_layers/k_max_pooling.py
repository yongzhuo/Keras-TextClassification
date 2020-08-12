# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/22 7:34
# @author   :Mo
# @function :


from keras.layers import Layer
import tensorflow as tf


class k_max_pooling(Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=8, **kwargs):
        self.top_k = top_k
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        inputs_reshape = tf.transpose(inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=inputs_reshape, k=self.top_k, sorted=False).values
        pool_top_k_reshape = tf.transpose(pool_top_k, perm=[0, 2, 1])
        return pool_top_k_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]
