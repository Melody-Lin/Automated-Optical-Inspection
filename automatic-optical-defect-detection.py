#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

# Input
TRAIN_PATH = '../input/aoioao/aoi/train_images/train_images/'
TEST_PATH = '../input/aoioao/aoi/test_images/test_images/'

train_csv = pd.read_csv('../input/aoioao/aoi/train.csv')
test_csv = pd.read_csv('../input/aoioao/aoi/test.csv')


# Shuffle
train_csv = train_csv.sample(frac=1).reset_index(drop=True)
num = int(train_csv.shape[0] * 0.8)
num2 = int(train_csv.shape[0] * 0.9)


# train data
train_data = []
train_label = []
for step in range(train_csv.shape[0]):
    image = cv2.imread(TRAIN_PATH+train_csv['ID'][step])
    # Normalization
    image = (image - image.mean()) / image.std()
    image = cv2.resize(image, (75,75), interpolation=cv2.INTER_AREA)
    train_data.append(np.array(image))
    
    label = []
    for k in range(6):
        if train_csv['Label'][step]==k:
            label.append(1)
        else:
            label.append(0)
    train_label.append(label)
    
    # 資料擴增
    if step < num:
        for i in range(-1, 2):
            train_data.append(np.array(cv2.flip(image, i)))
            train_label.append(label)
    
    if (step+1) % 1000 == 0:
        print("prepare train data " + str(step+1) + " / " + str(train_csv.shape[0]) + " success OwO")
print("prepare train data " + str(step+1) + " / " + str(train_csv.shape[0]) + " success OwO")
train_data = np.array(train_data)
train_label = np.array(train_label)


num *= 4
num2 = num + int((len(train_data)-num)/2)

train_ins = train_data[:num]
train_ous = train_label[:num]
valid_ins = train_data[num:num2]
valid_ous = train_label[num:num2]
test_ins = train_data[num2:]
test_ous = train_label[num2:]


# model - resNet
model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_tensor=None, input_shape=(75,75,3), classes=6)


model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    loss='categorical_crossentropy',metrics=['accuracy'])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracyx', patience = 10)
history = model.fit(train_ins, train_ous ,batch_size=32,
                    validation_data=(valid_ins, valid_ous),
                    epochs = 30, callbacks=[callback],
                    verbose = 2)

prediction = model.predict(test_ins)
target = test_ous

acc = 0
for i in range(len(prediction)):
    if np.argmax(prediction[i])==np.argmax(target[i]):
        acc += 1 
acc = acc / len(prediction)
print('ACC: {}'.format(acc))

train_data = []


# output - aoioao
test_data = []
submission = pd.read_csv('../input/aoioao/sample.csv')
for step in range(test_csv.shape[0]):
    image = cv2.imread(TEST_PATH+test_csv['ID'][step])
    image = (image - image.mean()) / image.std()
    image = cv2.resize(image, (75,75), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    test_data.append(image)
    if (step+1) % 1000 == 0:
        test_data = np.array(test_data)
        # prediction
        prediction = model.predict(test_data)
        #prediction = prediction * (train_max - train_min) + train_min
        #prediction = np.round(prediction)
        for i in range(len(prediction)):
            submission['Label'][step-999+i] = np.argmax(prediction[i])
        print("prepare test data " + str(step+1) + " / " + str(test_csv.shape[0]) + " success OxO")
        test_data = []

test_data = np.array(test_data)
# prediction
prediction = model.predict(test_data)
for i in range(len(prediction)):
    submission['Label'][10000+i] = np.argmax(prediction[i])
print("prepare test data " + str(step+1) + " / " + str(test_csv.shape[0]) + " success OxO")

submission.to_csv('./submission.csv', index = False)

