# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:18:36 2020

@author: User
"""


from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

from keras.models import Model

import matplotlib.pyplot as plt

import numpy as np
import scipy.io as scio

from sklearn import preprocessing   

min_max_scaler = preprocessing.MinMaxScaler()  


dataFile = "./data.mat";

zc = scio.loadmat(dataFile)



x_train=np.transpose( zc['zc_demo3'])

#y_train= np.transpose(zc['rimg2'])



x_test= np.transpose(zc['testdemo'])

#y_test =np.transpose(zc['testdemoy'])



#x_train = x_train.astype('float32')/ 255.

#y_train = y_train.astype('float32') /255.



x_test=min_max_scaler.fit_transform(x_test)  
x_train=min_max_scaler.fit_transform(x_train)  
#x_test = x_test.astype('float32') / 255.

#y_test = y_test.astype('float32') /255
x_train = np.reshape(x_train, (len(x_train), 16, 16, 1));

#    y_train = np.reshape(y_train, (len(y_train),64,64,1))

x_test = np.reshape(x_test, (len(x_test), 16, 16, 1));

#    y_test = np.reshape(y_test, (len(y_test),64,64,1))


noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(16,16,1))


x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)




x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(32,(3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                nb_epoch=100,
                batch_size=2,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))




decoded_imgs = autoencoder.predict(x_test_noisy)
 
 
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(16, 16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(16, 16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





















