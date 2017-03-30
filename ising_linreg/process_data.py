from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import pickle

from tensorflow.python.framework import dtypes



class DataSet(object):

	def __init__(self,data_X,data_Y,dtype=dtypes.float32):

		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

		assert data_X.shape[0] == data_Y.shape[0], ('data_X.shape: %s data_Y.shape: %s' % (data_X.shape, data_Y.shape))
		self._num_examples = data_X.shape[0]


		if dtype == dtypes.float32:
			data_X = data_X.astype(np.float32)
			data_X[np.where(data_X==0)]=-1 # replace 0 by -1
		self._data_X = data_X
		self._data_Y = data_Y #np.reshape(data_Y,(data_Y.shape[0],1))
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def data_X(self):
		return self._data_X

	@property
	def data_Y(self):
		return self._data_Y

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, seed=None):
		if seed:
			np.random.seed(seed)
		"""Return the next `batch_size` examples from this data set."""

		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._data_X = self._data_X[perm]
			self._data_Y = self._data_Y[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._data_X[start:end], self._data_Y[start:end]


def read_data_sets(data_params,dtype=dtypes.float32,train_size=80000,validation_size=5000):

	states = "mag_vs_T_L%i_T=%.2f.txt" %(data_params['L'],data_params['T'])
	energies = "energies_vs_T_L%i_T=%.2f.txt" %(data_params['L'],data_params['T'])
	
	Data=[]
	Data.append(np.loadtxt(states,delimiter=",",dtype=np.int))
	Data.append(np.loadtxt(energies,delimiter=",",dtype=np.int))


	# define test and train data sets
	train_data_X=Data[0][:train_size]
	train_data_Y=Data[1][:train_size]

	test_data_X=Data[0][train_size:]
	test_data_Y=Data[1][train_size:]

	if not 0 <= validation_size <= len(train_data_X):
		raise ValueError('Validation size should be between 0 and {}. Received: {}.'.format(len(train_data_X), validation_size))

	validation_data_X = train_data_X[:validation_size]
	validation_data_Y = train_data_Y[:validation_size]
	train_data_X = train_data_X[validation_size:]
	train_data_Y = train_data_Y[validation_size:]


	train = DataSet(train_data_X, train_data_Y, dtype=dtype)
	validation = DataSet(validation_data_X, validation_data_Y, dtype=dtype)
	test = DataSet(test_data_X, test_data_Y, dtype=dtype)

	Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

	return Datasets(train=train, validation=validation, test=test)


