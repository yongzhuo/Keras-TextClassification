# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/22 8:15
# @author   :Mo
# @function :


from keras.layers import Layer, Dense
import keras


class highway(Layer):
    """
    # writter by my own
    # paper； Highway Network(http://arxiv.org/abs/1505.00387).
    # 公式
    # 1. s = sigmoid(Wx + b)
    # 2. z = s * relu(Wx + b) + (1 - s) * x
    # x shape : [N * time_depth, sum(filters)]

    # Table 1. CIFAR-10 test set accuracy of convolutional highway networks with
    # rectified linear activation and sigmoid gates.
    # For comparison, results reported by Romero et al. (2014)
    # using maxout networks are also shown.
    # Fitnets were trained using a two step training procedure using soft targets from the trained Teacher network,
    # which was trained using backpropagation. We trained all highway networks directly using backpropagation.
    # * indicates networks which were trained only on a set of 40K out of 50K examples in the training set.



    # Figure 2. Visualization of certain internals of the blocks in the best 50 hidden layer highway networks trained on MNIST
    # (top row) and CIFAR-100 (bottom row). The first hidden layer is a plain layer which changes the dimensionality of the representation to 50. Each of
    # the 49 highway layers (y-axis) consists of 50 blocks (x-axis).
    # The first column shows the transform gate biases, which were initialized to -2 and -4 respectively.
    # In the second column the mean output of the transform gate over 10,000 training examples is depicted.
    # The third and forth columns show the output of the transform gates and
    # the block outputs for a single random training sample.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        gate_transform = Dense(units=K.int_shape(x)[1],
                               activation='sigmoid',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               bias_initializer=keras.initializers.Constant(value=-2))(x)
        gate_cross = 1 - gate_transform
        block_state = Dense(units=K.int_shape(x)[1],
                            activation='relu',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zero')(x)
        high_way = gate_transform * block_state + gate_cross * x
        return high_way

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[-1]
