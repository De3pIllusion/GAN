import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from matplotlib import pyplot as plt
import math
from PIL import Image
from tensorflow.keras import backend as K


def generator(inputs, image_size, activation='sigmoid', labels=None, codes=None):
    """generator model
    Arguments:
        inputs (layer): input layer of generator
        image_size (int): Target size of one side
        activation (string): name of output activation layer
        labels (tensor): input labels
        codes (list): 2-dim disentangled codes for infoGAN
    returns:
        model: generator model
    """
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]
    inputs = [inputs, labels] + codes
    x = keras.layers.concatenate(inputs, axis=1)

    x = keras.layers.Dense(image_resize * image_resize * layer_filters[0])(x)
    x = keras.layers.Reshape((image_resize, image_resize, layer_filters[0]))(x)
    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2DTranspose(filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding='same')(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return keras.Model(inputs, x, name='generator')



def discriminator(inputs, activation='sigmoid', num_labels=None, num_codes=None):
    """discriminator model
    Arguments:
        inputs (Layer): input layer of the discriminator
        activation (string): name of output activation layer
        num_labels (int): dimension of one-hot labels for ACGAN & InfoGAN
        num_codes (int): num_codes-dim 2 Q network if InfoGAN
    Returns:
        Model: Discriminator model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same')(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1)(x)
    if activation is not None:
        print(activation)
        outputs = keras.layers.Activation(activation)(outputs)
    if num_labels:
        layer = keras.layers.Dense(layer_filters[-2])(x)
        labels = keras.layers.Dense(num_labels)(layer)
        labels = keras.layers.Activation('softmax', name='label')(labels)
        # 1-dim continous Q of 1st c given x
        code1 = keras.layers.Dense(1)(layer)
        code1 = keras.layers.Activation('sigmoid', name='code1')(code1)
        # 1-dim continous Q of 2nd c given x
        code2 = keras.layers.Dense(1)(layer)
        code2 = keras.layers.Activation('sigmoid', name='code2')(code2)
        outputs = [outputs, labels, code1, code2]
    return keras.Model(inputs, outputs, name='discriminator')

