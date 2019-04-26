#!/usr/bin/python
# -*- coding:utf-8 -*-
#seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
from keras.utils import np_utils
import cPickle
from PIL import Image
nb_classes = 8
import numpy as np
import theano as th
import theano.tensor as T
np.random.seed(1337)

from sklearn import svm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, merge, ZeroPadding2D, Input, GlobalAveragePooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
early_stopping = EarlyStopping(monitor='val_acc', patience=15)
from keras import backend as K
K.set_image_dim_ordering('th')
import csv
# import cv2
import os
import glob

nb_epoch = 200

# emotion_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':5, 'Surprise':6, 'Neutral':4}
dictionary = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
emotion_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}

def conv_block(x, nb_filter, kernel_size=3, trainable=True):
	k = nb_filter

	out = ZeroPadding2D((1, 1))(x)
	out = Conv2D(k, (kernel_size, kernel_size), strides=(2, 2), trainable=trainable)(out)
	out = BatchNormalization(axis=1, trainable=trainable)(out)
	out = Activation('relu')(out)

	out = Conv2D(k, (kernel_size, kernel_size), padding='same', trainable=trainable)(out)
	out = BatchNormalization(axis=1, trainable=trainable)(out)
	# out = Activation('relu')(out)

	x = Conv2D(k, (1, 1), strides=(2, 2), trainable=trainable)(x)
	x = BatchNormalization(axis=1, trainable=trainable)(x)

	# out = merge([out, x], mode='sum')
	out = add([out,x])
	out = Activation('relu')(out)

	return out

def identity_block(x, nb_filter, kernel_size=3, trainable=True):
	k = nb_filter #

	out = Conv2D(k, (kernel_size, kernel_size), padding='same', trainable=trainable)(x)
	out = BatchNormalization(axis=1, trainable=trainable)(out)
	out = Activation('relu')(out)

	out = Conv2D(k, (kernel_size, kernel_size), padding='same', trainable=trainable)(out)
	out = BatchNormalization(axis=1, trainable=trainable)(out)
	# out = Activation('relu')(out)

	# out = merge([out, x], mode='sum')
	out = add([out,x])
	out = Activation('relu')(out)
	return out

def DeepID(X_train, Y_train, X_test, Y_test):
	trainable = False

	# inp = Input(shape=(1, 48, 48), name='input_1')

	# out = Conv2D(20, (4, 4), trainable=trainable, name='conv2d_1')(inp)
	# out = Activation('relu', name='activation_1')(out)
	# out = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(out)
	# out = Conv2D(40, (3, 3), trainable=trainable, name='conv2d_2')(out)
	# out = Activation('relu', name='activation_2')(out)
	# out = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name='max_pooling2d_2')(out)
	# out = Conv2D(60, (3, 3), trainable=trainable, name='conv2d_3')(out)
	# out = Activation('relu', name='activation_3')(out)
	# out = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(out)

	# layer2 = Conv2D(80, (2, 2), trainable=True, name='conv2d_4')(out)
	# layer2 = Activation('relu', name='activation_4')(layer2)

	# layer1 = Flatten(name='flatten_1')(out)
	# layer2 = Flatten(name='flatten_2')(layer2)
	# layer1 = Dense(160, trainable=True, name='dense_1')(layer1)
	# layer2 = Dense(160, trainable=True, name='dense_2')(layer2)
	
	# deepid_layer = merge([layer1, layer2], mode='sum', name='merge_1')

	# deepid_layer = Dropout(0.4, name='dropout_1')(deepid_layer)

	# # out = Dense(10575, activation='softmax', name='dense_3')(deepid_layer)
	# out = Dense(8, activation='softmax',name='fer_classes')(deepid_layer)
	# deepid_model = Model(inp, out)
	inp = Input(shape=(1, 48, 48))

	out = Conv2D(20, (4, 4), trainable=trainable)(inp)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(2, 2))(out)
	out = Conv2D(40, (3, 3), trainable=trainable)(out)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(out)
	out = Conv2D(60, (3, 3), trainable=trainable)(out)
	out = Activation('relu')(out)
	out = MaxPooling2D(pool_size=(2, 2))(out)

	layer2 = Conv2D(80, (2, 2), trainable=trainable)(out)
	layer2 = Activation('relu')(layer2)

	layer1 = Flatten()(out)
	layer2 = Flatten()(layer2)
	layer1 = Dense(160, trainable=trainable)(layer1)
	layer2 = Dense(160, trainable=trainable)(layer2)
	
	deepid_layer = add([layer1, layer2])

	deepid_layer = Dropout(0.4)(deepid_layer)

	# out = Dense(10575, activation='softmax', name='dense_3')(deepid_layer)
	out = Dense(8, activation='softmax', name='fer_classes')(deepid_layer)
	deepid_model = Model(inp, out)
	# deepid = Model(inp, deepid_layer)

	sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
	deepid_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# deepid_model.load_weights('./deepid_casia_weights9.hdf5', by_name=True)
	

	t2 = False
	res = ZeroPadding2D((1, 1))(inp)
	res = Conv2D(16, (3, 3), trainable=t2)(res)
	res = BatchNormalization(axis=1, trainable=t2)(res)
	res = Activation('relu')(res)

	res = identity_block(res,16,3,t2)
	res = identity_block(res,16,3,t2)
	res = identity_block(res,16,3,t2)
	
	res = conv_block(res,32,3,t2)
	res = identity_block(res,32,3,t2)
	res = identity_block(res,32,3,t2)
	
	res = conv_block(res,64,3,t2)
	res = identity_block(res,64,3,t2)
	res = identity_block(res,64,3,t2)

	avg = GlobalAveragePooling2D()(res)
	#
	res = Dense(8, activation='softmax')(avg)
	
	resnet_model = Model(inp, res)
	sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
	resnet_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# resnet_model.load('resnet_ferplus_weights.hdf5', by_name=True)
	# joint = merge([res,out], mode='sum')
	joint = concatenate([deepid_layer, avg], axis=1)
	# joint = BatchNormalization(axis=1)(joint)
	# joint = Dropout(0.2)(joint)
	joint = Dense(8, activation='softmax')(joint)
	model = Model(inp, joint)
	model.summary()
	sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.load_weights('resnet_ferplus_weights_joint2.hdf5')
	# resnet_model.load_weights('resnet_ferplus_weights_new3.hdf5')
	best_weights_filepath = './resnet_ferplus_weights_joint3.hdf5'
	# best_weights_filepath = './joint_weights.hdf5'

	saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

	datagen = ImageDataGenerator(
		rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
		horizontal_flip=True,
		fill_mode='nearest')

	# # compute quantities required for featurewise normalization
	# # (std, mean, and principal components if ZCA whitening is applied)
	'''
	datagen.fit(X_train)

	# # fit the model on the batches generated by datagen.flow()
	hist = model.fit_generator(datagen.flow(X_train, Y_train,
		batch_size=128),
		samples_per_epoch=X_train.shape[0],
		nb_epoch=nb_epoch,
		# validation_split=0.1,
		validation_data=(X_test, Y_test),
		callbacks=[saveBestModel])
	'''
	# model.summary()
	return model


def main():

	X_train = np.load('ferplus_train_x.npy')
	Y_train = np.load('ferplus_train_y.npy')
	X_val = np.load('ferplus_val_x.npy')
	Y_val = np.load('ferplus_val_y.npy')
	X_test = np.load('ferplus_test_x.npy')
	Y_test = np.load('ferplus_test_y.npy')

	X_train = X_train.astype('float32')
	X_train = (X_train - np.mean(X_train))/np.std(X_train)
	X_val = X_val.astype('float32')
	X_val = (X_val - np.mean(X_val))/np.std(X_val)
	X_test = X_test.astype('float32')
	X_test = (X_test - np.mean(X_test))/np.std(X_test)

	Y_train = np_utils.to_categorical(Y_train, 8)
	Y_val = np_utils.to_categorical(Y_val, 8)
	Y_test = np_utils.to_categorical(Y_test, 8)



	# X = np.load('ck_x_1308.npy')
	# print X.shape
	model = DeepID(X_train, Y_train, X_test, Y_test)
	pred = model.predict(X_test)
	cnt=0
	for i in xrange(X_test.shape[0]):
		if np.argmax(pred[i]) == np.argmax(Y_test[i]):
			cnt += 1
	print cnt * 1.0 / X_test.shape[0]
	print pred.shape
	print X_test.shape[0]
	np.save('joint_learning_pred.npy', pred)
	# X_train_id_features = deepid_layer.predict(X_train)
	# X_val_id_features = deepid_layer.predict(X_val)
	# X_test_id_features = deepid_layer.predict(X_test)

	# X = deepid_layer.predict(X)
	# print X.shape
	# np.save('ck_x_deepid.npy', X)
	# print X_train_id_features.shape
	# np.save('fer_train_deepid.npy', X_train_id_features)
	# np.save('fer_val_deepid.npy', X_val_id_features)
	# np.save('fer_test_deepid.npy', X_test_id_features)


	# print X_train_id_features.shape, X_val_id_features.shape, X_test_id_features.shape


if __name__ == '__main__':
	main()
