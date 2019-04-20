from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, Dense, Flatten, BatchNormalization, ReLU


__all__ = ['InceptionBlock1', 'InceptionBase', 'Inception']



class InceptionBlock1(Layer):
  """docstring for InceptionBlock"""
  def __init__(self, input_channels):
    super(InceptionBlock1, self).__init__()
    self.input_channels = input_channels
    self.conv_1x3 = Conv2D(filters=self.input_channels*2, kernel_size=[1, 3], padding='same')

    self.conv_3x1 = Conv2D(filters=self.input_channels*4, kernel_size=[3, 1], strides=2,
                            padding='same')

    self.conv_3x3 = Conv2D(filters=self.input_channels*4, kernel_size=[3, 3], strides=2,
                            padding='same')

    self.conv_1x1_b0 = Conv2D(filters=self.input_channels, kernel_size=[1, 1], padding='same')

    self.conv_1x1_b1 = Conv2D(filters=self.input_channels, kernel_size=[1, 1], padding='same')

    self.conv_1x1_b2 = Conv2D(filters=self.input_channels, kernel_size=[1, 1], padding='same')

    self.pooling = AveragePooling2D(strides=2, padding='same')

    self.batch_norm_b0 = BatchNormalization()

    self.batch_norm_b1 = BatchNormalization()

    self.batch_norm_b2 = BatchNormalization()

    self.activation = ReLU()



  def call(self, x):
    return self.wide_block(x)

  @tf.function
  def wide_block(self, input):
    branch_0 = self.conv_1x1_b0(input)
    branch_0 = self.conv_1x3(branch_0)
    branch_0 = self.conv_3x1(branch_0)
    branch_0 = self.activation(branch_0)
    branch_0 = self.batch_norm_b0(branch_0)

    branch_1 = self.conv_1x1_b1(input)
    branch_1 = self.conv_3x3(branch_1)
    branch_1 = self.activation(branch_1)
    branch_1 = self.batch_norm_b1(branch_1)

    branch_2 = self.pooling(input)
    branch_2 = self.conv_1x1_b2(branch_2)
    branch_2 = self.activation(branch_2)
    branch_2 = self.batch_norm_b2(branch_2)

    return tf.concat([branch_0, branch_1, branch_2], axis=3)



class InceptionBase(tf.keras.Model):
  """docstring for Model"""
  def __init__(self):
    super(InceptionBase, self).__init__()
    
    self.shortcut_1 = Conv2D(filters=144, strides=2, kernel_size=[1, 1], padding='same', kernel_initializer='zeros')
    self.shortcut_2 = Conv2D(filters=288, strides=2, kernel_size=[1, 1], padding='same', kernel_initializer='zeros')
    self.shortcut_3 = Conv2D(filters=576, strides=2, kernel_size=[1, 1], padding='same', kernel_initializer='zeros')
    self.shortcut_4 = Conv2D(filters=900, strides=2, kernel_size=[1, 1], padding='same', kernel_initializer='zeros')

    self.block_1 = InceptionBlock1(8)
    self.block_2 = InceptionBlock1(16)
    self.block_3 = InceptionBlock1(32)
    self.block_4 = InceptionBlock1(64)
    self.block_5 = InceptionBlock1(100)

  def call(self, input):
    input = self.block_1(input)
    shortcut_1 = self.shortcut_1(input)
    
    input = self.block_2(input)
    input = input + shortcut_1
    shortcut_2 = self.shortcut_2(input)
    
    input = self.block_3(input)
    input = input + shortcut_2
    shortcut_3 = self.shortcut_3(input)
    
    input = self.block_4(input)
    input = input + shortcut_3
    shortcut_4 = self.shortcut_4(input)
    
    return self.block_5(input) + shortcut_4



class Inception(tf.keras.Model):
  def __init__(self, num_label):
    super(Inception, self).__init__()
    self.extractor = InceptionBase()
    self.flatten = Flatten()
    self.pooling = AveragePooling2D(pool_size=[3, 3], strides=2, padding='same')
    self.logit_layer = Dense(num_label, activation='softmax')
   
  def call(self, input):
    logit = self.extractor(input)
    logit = self.pooling(logit)
    logit = self.flatten(logit)
    return self.logit_layer(logit)


