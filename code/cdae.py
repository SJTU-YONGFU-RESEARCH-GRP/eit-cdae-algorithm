# -*- coding: utf-8 -*-
"""
@author: Yongfu Li
"""
import keras
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as scio
 
dataFile = "./data.mat";
zc = scio.loadmat(dataFile)
epochs = 1000

############
## Option ##
############
options = 1; # 0: sae 1: encoder
encoder_option = 2;
decoder_option = 2;

#######################################
if options == 0:
    input_img = Input(shape=(1,256))
    #############
    ## Encoder ##
    #############
    encoded = Dense(input_dim=256, output_dim=150, activation='relu')(input_img)
    encoded = Dense(input_dim=150, output_dim=150, activation='relu')(encoded)
    decoded = Dense(input_dim=150, output_dim=4096, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
if options == 1:
    input_img = Input(shape=(16,16,1))
    #############
    ## Encoder ##
    #############
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

    if encoder_option == 1:
        pass;
    if encoder_option == 2:
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    ##################
    ## Hidden Layer ##
    ##################
    # at this point the representation is (2, 2, 8), i.e. 128-dimensional

    ###################
    ## Decoder Layer ##
    ###################
    if decoder_option == 1:
        #x = UpSampling2D((2, 2))(encoded)
        #x = UpSampling2D((2, 2))(x)
        #x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    if decoder_option == 2:
        #x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        #x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        #x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid',padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # meansquareerror, 

# To train it, use the original MNIST digits with shape (samples, 3, 28, 28),
# and just normalize pixel values between 0 and 1

#(x_train, _), (x_test, _) = mnist.load_data()

x_train=np.transpose( zc['zc_demo3'])
y_train= np.transpose(zc['rimg2'])

x_test= np.transpose(zc['testdemo'])
y_test =np.transpose(zc['testdemoy'])

x_train = x_train.astype('float32')/ 255.
y_train = y_train.astype('float32') /255.


x_test = x_test.astype('float32') / 255.
y_test = y_test.astype('float32') /255

if options == 0:
    x_train = np.reshape(x_train, (len(x_train), 1, 256));
    y_train = np.reshape(y_train, (len(y_train), 1, 4096))
    x_test = np.reshape(x_test, (len(x_test), 1, 256));
    y_test = np.reshape(y_test, (len(y_test), 1, 4096))
if options == 1:
    x_train = np.reshape(x_train, (len(x_train), 16, 16, 1));
    y_train = np.reshape(y_train, (len(y_train),64,64,1))
    x_test = np.reshape(x_test, (len(x_test), 16, 16, 1));
    y_test = np.reshape(y_test, (len(y_test),64,64,1))

# open a terminal and start TensorBoard to read logs in the autoencoder subdirectory

# tensorboard --logdir=autoencoder


#autoencoder.fit(x_train, x_train, epochs=1, batch_size=11, shuffle=True, validation_data=(x_test, x_test),               
#                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)

autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=11, shuffle=True, validation_data=(x_test, y_test),               
                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)


decoded_imgs = autoencoder.predict(x_test)



n =5
plt.set_cmap('gray')

plt.figure(dpi=130)
ax = plt.subplot(2, n,  1)  
plt.imshow(y_test[11].reshape(64, 64))
ax.set_axis_off()
ax = plt.subplot(2, n,  6) 
plt.imshow(decoded_imgs[11].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  2)  
plt.imshow(y_test[22].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  3)  
plt.imshow(y_test[33].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  4)  
plt.imshow(y_test[44].reshape(64, 64))
ax.set_axis_off()
ax = plt.subplot(2, n,  5)  
plt.imshow(y_test[55].reshape(64, 64))
ax.set_axis_off()
ax = plt.subplot(2, n,  10) 
plt.imshow(decoded_imgs[55].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  7) 
plt.imshow(decoded_imgs[22].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  8) 
plt.imshow(decoded_imgs[33].reshape(64, 64))
ax.set_axis_off()

ax = plt.subplot(2, n,  9) 
plt.imshow(decoded_imgs[44].reshape(64, 64))
ax.set_axis_off()
'''
for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

#    plt.imshow(x_test[i].reshape(28, 28))


    plt.imshow(y_test[i].reshape(64, 64))
    
    plt.gray()

    ax.set_axis_off()



    # display reconstruction

    ax = plt.subplot(2, n, i + n + 1)

#    plt.imshow(decoded_imgs[i].reshape(28, 28))
    
    plt.imshow(decoded_imgs[i].reshape(64, 64))

    plt.gray()

    ax.set_axis_off()

'''

plt.show()

plt.savefig('image.png')

# take a look at the 128-dimensional encoded representation

# these representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images



encoder = Model(input_img, encoded)

encoded_imgs = encoder.predict(x_test)



# save latent space features 128-d vector

pickle.dump(encoded_imgs, open('conv_autoe_features.pickle', 'wb'))


#
#n = 10
#
#plt.figure(figsize=(10, 4), dpi=100)
#
#for i in range(n):
#
#    ax = plt.subplot(1, n, i + 1)
#
#    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
#
#    plt.gray()
#
#    ax.set_axis_off()



plt.show()



K.clear_session()
