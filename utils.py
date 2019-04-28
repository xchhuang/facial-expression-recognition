#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from keras.utils import np_utils
import params

def loadData(folder):

	nb_classes = params.nb_classes

	X_train = np.load(folder+'ferplus_train_x.npy')
	y_train = np.load(folder+'ferplus_train_y.npy')
	X_val = np.load(folder+'ferplus_val_x.npy')
	y_val = np.load(folder+'ferplus_val_y.npy')
	X_test = np.load(folder+'ferplus_test_x.npy')
	y_test = np.load(folder+'ferplus_test_y.npy')

	X_train = X_train.astype('float32')
	X_train = (X_train - np.mean(X_train))/np.std(X_train)
	X_val = X_val.astype('float32')
	X_val = (X_val - np.mean(X_val))/np.std(X_val)
	X_test = X_test.astype('float32')
	X_test = (X_test - np.mean(X_test))/np.std(X_test)
	
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_val = np_utils.to_categorical(y_val, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	
	return X_train, Y_train, X_val, Y_val, X_test, Y_test


def evaluate(pred, label):
	cnt = 0
	n = len(label)
	for i in range(n):
		if np.argmax(pred[i]) == np.argmax(label[i]):
			cnt += 1
	return cnt * 1.0 / n