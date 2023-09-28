# coding: utf-8

# In[ ]:


import os
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import ast
import numpy as np
import pandas as pd

os.chdir('train')  # direct

num_train_virus = 99815
num_train_host = 96963
num_test_virus = 99892
num_test_host = 99997

true_num_train_virus = 0
true_num_train_host = 0
true_num_test_virus = 0
true_num_test_host = 0

# construct training dataset
m = []
g = open('train-sample.csv', 'r')
lines = g.readlines()
for count, line in enumerate(lines):
    n = line[:-1]  # remove LF character
    if n == "":
        continue
    try:
        k = ast.literal_eval(n)
        j = list(k)
        m.append(j)
        if count < num_train_virus:
            true_num_train_virus += 1
        else:
            true_num_train_host += 1
    except ValueError:
        print("Can't evaluate literal", flush=False)
    except TypeError:
        print("Can't iterate literal", flush=False)
g.close()
X_train = m
num_train_virus = true_num_train_virus
num_train_host = true_num_train_host
assert len(X_train) == num_train_virus + num_train_host

# construct testing dataset
a = []
f = open('test-sample.csv', 'r')
lines = f.readlines()
for count, line in enumerate(lines):
    b = line[:-1]  # remove LF character
    if b == "":
        continue
    try:
        c = ast.literal_eval(b)
        d = list(c)
        a.append(d)
        if count < num_test_virus:
            true_num_test_virus += 1
        else:
            true_num_test_host += 1
    except ValueError:
        print("Can't evaluate literal", flush=False)
    except TypeError:
        print("Can't iterate literal", flush=False)
f.close()
X_test = a
num_test_virus = true_num_test_virus
num_test_host = true_num_test_host
assert len(X_test) == num_test_virus + num_test_host

# construct testing labels
y_train = []
for i in range(num_train_virus):
    y_train.append(1)
for i in range(num_train_host):
    y_train.append(0)

# construct training labels
y_test = []
for i in range(num_test_virus):
    y_test.append(1)
for i in range(num_test_host):
    y_test.append(0)

# vocabulary length
VOCAB_LEN = 65
# max sequence length
SEQUENCE_LEN = 500
# padding
X_train = pad_sequences(X_train, maxlen=SEQUENCE_LEN, value=0.)
X_test = pad_sequences(X_test, maxlen=SEQUENCE_LEN, value=0.)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# embedding size
WORD_FEATURE_DIM = 20

DOC_FEATURE_DIM = 64
net = tflearn.input_data([None, SEQUENCE_LEN])
net = tflearn.embedding(net, input_dim=VOCAB_LEN, output_dim=WORD_FEATURE_DIM)
net = tflearn.lstm(net, DOC_FEATURE_DIM, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0002, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="/tmp/tflearn_logs/")

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# training
model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True, n_epoch=25, batch_size=256)
