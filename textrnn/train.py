# -*- coding: utf-8 -*-
'''
  @CreateTime	:  2021/12/07 13:38:59
  @Author	:  Alwin Zhang
  @Mail	:  zjfeng@homaytech.com
'''


import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import sys
import os
sys.path.append("./")
from textrnn.model import TextRNN


np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

class_num = 2
max_len = 400
embedding_dims = 100
epochs = 1
batch_size = 8

max_features = 5000

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices(
#     (x_test, y_test)).batch(batch_size)

print('Build model...')

# model = TextRNN(max_len=max_len, max_features=max_features,
#                 embedding_dims=embedding_dims, class_num=class_num, last_activation='softmax')

# model with feedforward attention
model = TextRNN(max_len=max_len, max_features=max_features,
                     embedding_dims=embedding_dims, class_num=class_num, last_activation='softmax', attn=True)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.001)


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

model.build((None,400))
model.summary()

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, callbacks=[EarlyStoppingAtMinLoss()])

loss, acc = model.evaluate(x_test, y_test)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
