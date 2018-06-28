import numpy as np
import tensorflow as tf
import gc
import time
import pickle
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import random

labels = pd.read_csv('./labels/labels.csv')
ylabel = labels.breed
oh = pd.get_dummies(ylabel)
ohv = oh.values

train = ZipFile('/home/kaleeswaran/Desktop/Capstone/train.zip', 'r')

'''
for i in range(int((len(train.namelist()) - 1)/32) + 1):
    tin = time.time()
    namelist = train.namelist()[i*32+1:(i+1)*32+1]
    images   = np.zeros((len(namelist), 90, 90, 3))
    for j in range(len(namelist)):
        filename = BytesIO(train.read(namelist[j]))
        image = PIL.Image.open(filename)
        image = image.resize((90, 90))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0)
        images[j] = image
    pickle.dump(images, open('./batch_pickles/batchpickles' + str(i+1) + '.p', 'wb'))
    print('iteration ' + str(i) + ' time: ' + str(time.time() - tin))
    gc.collect()
'''

test = []
for i in range(300,321):
    ij = pickle.load(open('./batch_pickles/batchpickles' + str(i) + '.p', 'rb'))
    test.append(ij)

test = np.concatenate(test, 0)

def augmentation_func(imgvec):
    imgaug = np.zeros(imgvec.shape)
    for i in range(imgvec.shape[0]):
        toss = random.randint(0,9)
        if toss == 0:
            imgaug[i] = imgvec[i]
        else:
            tossed = random.randint(0,5)
            if tossed == 0:
                M   = cv2.getRotationMatrix2D((45, 45),30,1)
                img = cv2.warpAffine(imgvec[i],M,(90, 90))
                imgaug[i] = img
            elif tosses == 1:
                M   = np.float32([[1, 0, 10],[0, 1, 10]])
                img = cv2.warpAffine(imgvec[i], M, (90, 90))
                imgaug[i] = img
            elif tosses == 2:
                pts1 = np.float32([[20,20],[70,20],[20,70]])
                pts2 = np.float32([[20-15,20+0],[70-15,20-0],[20+15,70-0]])
                M = cv2.getAffineTransform(pts1,pts2)
                img = cv2.warpAffine(imgvec[i],M,(100,100))
                imgaug[i] = img
            elif tosses == 3:
                img = cv2.flip(imgvec[i], 0)
                imgaug[i] = img
            else:
                img = cv2.flip(imgvec[i], 1)
                imgaug[i] = img
        return()

def conv_layer(x, shape, num):
    w = tf.get_variable('w' + str(num), shape = shape, initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b = tf.zeros(shape[-1], name='b' + str(num))
    conv = tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")
    conv = tf.nn.bias_add(conv, b)
    return(tf.nn.relu(conv))

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 90, 90, 3))
Y = tf.placeholder(tf.float32, shape=(None, 120))

conv1 = conv_layer(X, [3,3,3,32], 1) #?,90,90,8
conv2 = conv_layer(conv1, [3,3,32,32], 2) #?90,90,8
maxp1 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='SAME')

conv3 = conv_layer(maxp1, [5,5,32,64], 3) #?,45,45,16
conv4 = conv_layer(conv3, [5,5,64,64], 4) #?,45,45,16
maxp2 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding='SAME')

fc    = tf.contrib.layers.flatten(maxp2)

fc1   = tf.contrib.layers.fully_connected(fc, 4096, activation_fn=None)

bn2   = tf.layers.batch_normalization(fc1)

Fc1   = tf.nn.relu(bn2)

fc2   = tf.contrib.layers.fully_connected(Fc1, 2048)

fc3   = tf.contrib.layers.fully_connected(fc2, 120, activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_test  = test
y_train = ohv[:299*32]
y_test  = ohv[299*32:]

init = tf.global_variables_initializer()
trc = []
tec = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        tin = time.time()
        z = 0
        for batch in np.repeat(np.arange(280),2):
            z += 1
            if z%2 == 1:
                imvec = pickle.load(open('./batch_pickles/batchpickles' + str(batch + 1) + '.p', 'rb'))
                _, trcost, tracc = sess.run([optimizer, cost, accuracy], feed_dict = {X: imvec, Y: y_train[batch*32:(batch+1)*32]})
                print('in')
                tecost, teacc = sess.run([cost, accuracy], feed_dict = {X: x_test[:250], Y: y_test[:250]})
                trc.append(trcost)
                tec.append(tecost)
                print('epoch: ' + str(epoch) + ' batch: ' + str(batch))
                print('train cost: ' + str(trcost))
                print('train accuracy: ' + str(tracc))
                print('test cost: ' + str(tecost))
                print('test accuracy: ' + str(teacc))
            else:
                imvec = pickle.load(open('./batch_pickles/batchpickles' + str(batch + 1) + '.p', 'rb'))
                imvec = augmentation_func(imvec)
                _, trcost, tracc = sess.run([optimizer, cost, accuracy], feed_dict = {X: imvec, Y: y_train[batch*32:(batch+1)*32]})
                print('in')
                tecost, teacc = sess.run([cost, accuracy], feed_dict = {X: x_test[:250], Y: y_test[:250]})
                trc.append(trcost)
                tec.append(tecost)
                print('epoch: ' + str(epoch) + ' batch: ' + str(batch))
                print('train cost: ' + str(trcost))
                print('train accuracy: ' + str(tracc))
                print('test cost: ' + str(tecost))
                print('test accuracy: ' + str(teacc))
        print('time: ' + str(time.time() - tin))
