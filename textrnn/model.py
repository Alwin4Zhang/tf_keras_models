from textrnn.attention import Attention
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import os
import sys
sys.path.append("./")


def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn


class TextRNN(Model):
    def __init__(self, max_len, max_features, embedding_dims, class_num, last_activation='softmax', dense_size=None, attn=None):
        super(TextRNN, self).__init__()
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size
        self.attn = attn
        if self.attn:
            self.attn_layer = Attention()

        self.embedding = Embedding(
            input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.max_len)
        self.rnn = Bidirectional(layer=GRU(
            units=128, activation='tanh', return_sequences=True), merge_mode='concat')  # LSTM or GRU
        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(
            self.class_num, activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        assert len(inputs.get_shape()) == 2
        assert inputs.get_shape()[1] == self.max_len

        emb = self.embedding(inputs)
        x = self.rnn(emb)
        if self.attn:
            x = self.attn_layer(x)
        else:
            x = tf.reduce_mean(x, axis=1)
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)

    def get_config(self):
        return {
            "rnn": self.rnn
        }


if __name__ == '__main__':
    model = TextRNN(
        max_len=400,
        max_features=5000,
        embedding_dims=100,
        class_num=2,
        last_activation='softmax'
    )
    model.build_graph(input_shape=(None, 400))
    model.summary()
    config = model.get_config()
    print(config)
