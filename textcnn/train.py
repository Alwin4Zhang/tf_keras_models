# -*- coding: utf-8 -*-
'''
  @CreateTime	:  2021/12/07 13:38:59
  @Author	:  Alwin Zhang
  @Mail	:  zjfeng@homaytech.com
'''
import sys
import os

sys.path.append("./")
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textcnn.model import TextCNN


np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

class_num = 2
max_len = 400
embedding_dims = 100
epochs = 10
batch_size = 6

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

model = TextCNN(max_len=max_len, max_features=max_features,
                embedding_dims=embedding_dims, class_num=class_num, kernel_sizes=[2, 3, 5], kernel_regularizer=None, last_activation='softmax')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.001)

model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=["accuracy"]
)

model.fit(x_train,y_train)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)




# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#     name='train_accuracy')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#     name='test_accuracy')

# # 使用tf.GradientTape 来训练模型


# @tf.function
# def train_step(x, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(x)
#         loss = loss_fn(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     train_loss(loss)
#     train_accuracy(labels, predictions)

# # 测试模型：


# @tf.function
# def test_step(x, labels):
#     predictions = model(x)
#     t_loss = loss_object(labels, predictions)

#     test_loss(t_loss)
#     test_accuracy(labels, predictions)


# def inference(x, model=None):
#     predictions = model(x)
#     return np.argmax(predictions, axis=-1)


# print("Train...")

# for epoch in range(epochs):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()

#     for x, labels in train_ds:
#         train_step(x, labels)

#     for x, test_labels in test_ds:
#         test_step(x, test_labels)
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print(template.format(epoch+1,
#                           train_loss.result(),
#                           train_accuracy.result()*100,
#                           test_loss.result(),
#                           test_accuracy.result()*100))

# print("Test...")
# pred = np.array([])
# for x, test_labels in test_ds:
#     pred = np.append(pred, inference(x))
# print("pred is : ", pred)
