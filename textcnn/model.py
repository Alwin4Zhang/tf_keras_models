# -*- coding: utf-8 -*-
'''
  @CreateTime	:  2021/12/07 12:43:25
  @Author	:  Alwin Zhang
  @Mail	:  zjfeng@homaytech.com
'''

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model


class TextCNN(Model):
    def __init__(self, max_len, max_features, embedding_dims, class_num, kernel_sizes=[1, 2, 3], kernel_regularizer=None, last_activation='softmax'):
        """
        :param max_len:文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param class_num:
        :param last_activation:
        """
        super(TextCNN, self).__init__()
        self.max_len = max_len
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = Embedding(
            input_dim=max_features, output_dim=embedding_dims, input_length=max_len)
        self.conv1s = []
        self.avgpools = []
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size,
                               activation='relu', kernel_regularizer=kernel_regularizer))
            self.avgpools.append(GlobalMaxPooling1D())
        self.classifier = Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        assert len(inputs.get_shape()) == 2
        assert inputs.get_shape()[1] == self.max_len

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb)
            c = self.avgpools[i](c)
            conv1s.append(c)
        x = Concatenate()(conv1s)  # batch_size,len(self.kernel_sizes) * filters
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)