import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from zipfile import ZipFile
from io import BytesIO
import PIL.Image
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time
import gc
from sys import getsizeof

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=True)

weights = conv_base.get_weights()

pickle.dump(weights, open('/home/kaleeswaran/Desktop/Capstone/WeightsTop.p', 'wb'))
vgg16wt = pickle.load(open('/home/kaleeswaran/Desktop/Capstone/WeightsTop.p', 'rb'))

tf.get_default_graph()

img = tf.placeholder(dtype="float32", shape=(None, 224, 224, 3))

def convolution(im, w, b):
    conv = tf.nn.conv2d(im, w, [1,1,1,1], padding="SAME")
    conv = tf.nn.bias_add(conv, b)
    return(tf.nn.relu(conv))

def maxpool(im):
    return(tf.nn.max_pool(im, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"))

conv_1 = convolution(img, vgg16wt[0], vgg16wt[1])
conv_2 = convolution(conv_1, vgg16wt[2], vgg16wt[3])
maxp_1 = maxpool(conv_2)

conv_3 = convolution(maxp_1, vgg16wt[4], vgg16wt[5])
conv_4 = convolution(conv_3, vgg16wt[6], vgg16wt[7])
maxp_2 = maxpool(conv_4)

conv_5 = convolution(maxp_2, vgg16wt[8], vgg16wt[9])
conv_6 = convolution(conv_5, vgg16wt[10], vgg16wt[11])
conv_7 = convolution(conv_6, vgg16wt[12], vgg16wt[13])
maxp_3 = maxpool(conv_7)

conv_8 = convolution(maxp_3, vgg16wt[14], vgg16wt[15])
conv_9 = convolution(conv_8, vgg16wt[16], vgg16wt[17])
conv_10 = convolution(conv_9, vgg16wt[18], vgg16wt[19])
maxp_4 = maxpool(conv_10)

conv_11 = convolution(maxp_4, vgg16wt[20], vgg16wt[21])
conv_12 = convolution(conv_11, vgg16wt[22], vgg16wt[23])
conv_13 = convolution(conv_12, vgg16wt[24], vgg16wt[25])
maxp_5 = maxpool(conv_13)

fc = tf.contrib.layers.flatten(maxp_5)
FC = tf.nn.bias_add(tf.matmul(fc, vgg16wt[26]), vgg16wt[27])
FC = tf.nn.relu(FC)
F1 = tf.nn.bias_add(tf.matmul(FC, vgg16wt[28]), vgg16wt[29])
#F1 = tf.nn.relu(F1)
#F2 = tf.nn.bias_add(tf.matmul(F1, vgg16wt[30]), vgg16wt[31])

train = ZipFile('/home/kaleeswaran/Desktop/Capstone/train.zip', 'r')

with tf.Session() as sess:
    imagevec = np.zeros((10222, 4096))
    tin = time.time()
    for j in range(10222):
        if j%100 == 0:
            print('iteration ' + str(j) + ': ' + str(time.time() - tin))
            tin = time.time()
        filename = BytesIO(train.read(train.namelist()[j+1]))
        image = PIL.Image.open(filename)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0)
        imagevec[j] = sess.run(F1, feed_dict = {img: image.reshape(1, 224, 224, 3)})

pickle.dump(imagevec, open('/home/kaleeswaran/Desktop/Capstone/IVT.p', 'wb'))
IVT = pickle.load(open('/home/kaleeswaran/Desktop/Capstone/IVT.p', 'rb'))

labels = pd.read_csv('/home/kaleeswaran/Desktop/Capstone/labels/labels.csv')
ylabel = labels.breed
oh = pd.get_dummies(ylabel)
y_label = np.dot(oh.values, np.arange(1,121))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(IVT, ylabel, test_size = 0.2, random_state=1)

from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

from sklearn.decomposition import PCA

Peeca = PCA().fit(X_train)
plt.plot(np.cumsum(Peeca.explained_variance_ratio_))
plt.show()

pca = PCA(n_components=120)
pca.fit(X_train)

X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)    

clf1 = SVC()
clf1.fit(X_t_train, y_train)
clf1.score(X_t_train, y_train)

clf1.score(X_t_test, y_test)

from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(X_t_train, y_train)

clf2.score(X_t_train, y_train)
clf2.score(X_t_test, y_test)

