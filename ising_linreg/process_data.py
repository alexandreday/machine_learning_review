from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import collections
import pickle

np.random.seed(12)

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
		self._data_X = data_X
		self._data_Y = np.reshape(data_Y,(data_Y.shape[0],1))
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
		if seed is not None:
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


def ising_energies(states,Lx,Ly,noise_width=0.0):
	"""
	This function calculates the energies of the states in the nn Hamiltonian
	"""
	J=np.zeros((Lx,Ly,Lx,Ly),)
	for i in range(Lx):
		for j in range(Ly):
			for kl in [[1,0],[0,1]]: #[[0,-1],[1,0],[0,1],[-1,0]]:
					J[i,j,(i+kl[0])%Lx,(j+kl[1])%Ly]-=1.0
	J=J.reshape(Lx*Ly,Lx*Ly)
	# compute energies
	E = np.einsum('...i,ij,...j->...',states,J,states) + np.random.normal(0,noise_width,size=states.shape[0])
	# extract indices of nn interactions
	J=J.reshape(Lx*Ly,Lx*Ly)
	J=J.reshape(Lx*Ly*Lx*Ly,)
	J_sp = sp.csr_matrix(J)
	inds_nn=J_sp.nonzero()[1]

	return E, inds_nn


def ising_energies_1D(states,L,noise_width=0.0):
	"""
	This function calculates the energies of the states in the nn Hamiltonian
	"""
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L]-=0.5
		J[i,(i-1)%L]-=0.5
	# compute energies
	E = np.einsum('...i,ij,...j->...',states,J,states) + np.random.normal(0,noise_width,size=states.shape[0])
	# extract indices of nn interactions
	J=J.reshape(L*L,)
	J_sp = sp.csr_matrix(J)
	inds_nn=J_sp.nonzero()[1]

	return E, inds_nn

def unique_rows(a, **kwargs):

    rowtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    b = np.ascontiguousarray(a).view(rowtype)
    return_index = kwargs.pop('return_index', False)
    out = np.unique(b, return_index=True, **kwargs)
    idx = out[1]
    uvals = a[idx]
    if (not return_index) and (len(out) == 2):
        return uvals
    elif return_index:
        return (uvals,) + out[1:]
    else:
        return (uvals,) + out[2:]

def read_data_sets(data_params,dtype=dtypes.float32,train_size=80000,validation_size=0,noise_width=0.0):

	"""
	states_str = "mag_vs_T_L%i_T=%.2f.txt" %(data_params['L'],data_params['T'])
	
	states=np.loadtxt(states_str,delimiter=",",dtype=np.int)
	states[np.where(states==0)]=-1 # replace 0 by -1

	energies,inds_nn=ising_energies(states,data_params['L'],data_params['L'],noise_width=noise_width)
	"""

	states=np.random.choice([-1, 1], size=(400,data_params['L']))
	#print(states.shape )
	#exit()
	energies,inds_nn=ising_energies_1D(states,data_params['L'],noise_width=noise_width)

	states=np.einsum('...i,...j->...ij', states, states)
	shape=states.shape
	
	states=states.reshape((shape[0],shape[1]*shape[2]))

	#states, inds=unique_rows( states ,**{'return_index':True})
	#energies=energies[inds]


	# nearest neighbours only
	if data_params['nn']:
		states=states[:,inds_nn]
	
	Data=[states,energies]

	"""
	np.savetxt('ising_states_L={}_T={}.txt'.format(data_params['L'],data_params['T']),states)
	np.savetxt('ising_energies_L={}_T={}.txt'.format(data_params['L'],data_params['T']),energies)

	print('exiting...')
	"""

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


