#!/usr/bin/python
# -*- coding:utf-8 -*-

import params
from utils import *
from models import ResNet
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

	params.print_params()
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
	parser.add_argument('--aug', type=str2bool, default=False, help='source domain id')
	args = parser.parse_args()

	X_train, Y_train, X_val, Y_val, X_test, Y_test = loadData(params.data_folder)
	print (X_train.shape, Y_train.shape)
	print (X_val.shape, Y_val.shape)
	print (X_test.shape, Y_test.shape)

	model, avg_layer = ResNet(X_train, Y_train, X_test, Y_test, args)
	pred = model.predict(X_test)
	acc = evaluate(pred, Y_test)
	print ('\nAccuracy: {:.4f}'.format(acc))

if __name__ == '__main__':
	main()


