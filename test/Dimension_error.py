# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/11 22:57
# @author   :Mo
# @function :

from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate
from keras.models import Model
import keras.backend as K

"""This is the "inception" module."""
def incepm_v1(out_filters, input_shape)->Model:
    input_img = Input(shape=input_shape)

    tower_1 = Conv2D(out_filters, (1, 1), padding='same',
        activation='relu')(input_img)
    tower_1 = Conv2D(out_filters, (3, 3), padding='same',
        activation='relu')(tower_1)

    tower_2 = Conv2D(out_filters, (1, 1), padding='same',
        activation='relu')(input_img)
    tower_2 = Conv2D(out_filters, (5, 5), padding='same',
        activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(out_filters, (1, 1), padding='same',
        activation='relu')(tower_3)

    output = Concatenate(axis=1)([tower_1, tower_2, tower_3])

    model = Model(inputs=input_img, outputs=output)
    return model

"""This is then used in the following model"""
def Unetish_model1(image_shape=(3000, 3000, 3)):
    image = Input(shape=image_shape)

    #First layer 96X96
    conv1 = Conv2D(32, (3,3),padding='same', activation = 'relu')(image)
    conv1out = Conv2D(16, (1,1),padding = 'same', activation =
    'relu')(conv1)
    conv1out = MaxPooling2D((2,2), strides = (2,2))(conv1out)
    aux1out = Conv2D(16, (1,1), padding = 'same', activation  = 'relu')(conv1)

    #Second layer 48x48
    #conv2 = incepm_v1(64, conv1out.shape[1:])(conv1out)
    conv2 = incepm_v1(64, K.int_shape(conv1out)[1:])(conv1out)
    conv2out = Conv2D(32, (1,1), padding = 'same', activation =
        'relu')(conv2)
    conv2out = MaxPooling2D((2,2), strides = (2,2))(conv2out)
    aux2out = Conv2D(32, (1,1), padding = 'same', activation  =
        'relu')(conv2)

    #".... removed for sparsity"
    model = Model(inputs =image, outputs = aux2out)
    model.summary()
    return model

IMAGE_SIZE = 96
Unet = Unetish_model1(image_shape=(3000, 3000, 3))