# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import sys, os, argparse
import tensorflow as tf

#import process_data
#from linreg import Linear_Regression

# suppress tflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

##########################################
from tensorflow.python.framework import dtypes
import pickle


class DataSet(object):

	def __init__(self,data_X,data_Y,dtype=dtypes.float32):

		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid dtype %r, expected uint8 or float32' % dtype)

		assert data_X.shape[0] == data_Y.shape[0], ('data_X.shape: %s data_Y.shape: %s' % (data_X.shape, data_Y.shape))
		self.num_examples = data_X.shape[0]


		if dtype == dtypes.float32:
			# Convert from [0, 255] -> [0.0, 1.0].
			data_X = data_X.astype(np.float32)
			data_X = np.multiply(data_X+4.0, 1.0 / 8.0)
			#data_X = np.multiply(data_X, 1.0 / 4.0)
		self.data_X = data_X
		self.data_Y = data_Y #np.reshape(data_Y,(data_Y.shape[0],30))
		
		self.epochs_completed = 0
		self.index_in_epoch = 0

	def next_batch(self, batch_size, seed=None):
		if seed:
			np.random.seed(seed)
		"""Return the next `batch_size` examples from this data set."""

		start = self.index_in_epoch
		self.index_in_epoch += batch_size
		if self.index_in_epoch > self.num_examples:
			# Finished epoch
			self.epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self.num_examples)
			np.random.shuffle(perm)
			self.data_X = self.data_X[perm]
			self.data_Y = self.data_Y[perm]
			# Start next epoch
			start = 0
			self.index_in_epoch = batch_size
			assert batch_size <= self.num_examples
		end = self.index_in_epoch
		return self.data_X[start:end], self.data_Y[start:end]


def read_data_sets(pkl_file, root_dir, dtype=dtypes.float32,train_size=80000,validation_size=5000):

	import collections

	L=40 # linear system size
	T=np.linspace(0.25,4.0,16) # temperatures
	T_c=2.26 # critical temperature in the TD limit

	# preallocate ordered, critical and disordered states
	X_ordered=np.zeros((90000,L**2),dtype=np.int32)
	Y_ordered=np.zeros((90000,2),dtype=np.int32)
	Y_ordered[:,1]=1

	X_disordered=np.zeros((80000,L**2),dtype=np.int32)
	Y_disordered=np.zeros((80000,2),dtype=np.int32)
	Y_ordered[:,0]=1

	# load data in ordered phase
	for i,T_i in enumerate(T[:9]):

		file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
		
		f=open(root_dir+pkl_file,'rb')
		X_ordered[10000*i:10000*(i+1),:]=np.unpackbits(pickle.load(f)).reshape(-1, L**2)


	# load data in disordered phase
	for i,T_i in enumerate(T[9:]):

		file_name='mag_vs_T_L%i_T=%.2f.pkl' %(L,T_i)
		
		f=open(root_dir+pkl_file,'rb')
		X_disordered[10000*i:10000*(i+1),:]=np.unpackbits(pickle.load(f)).reshape(-1, L**2)

	# map 0 state to -1 (Ising variable can take values $\pm 1$)
	X_ordered[np.where(X_ordered==0)]=-1
	X_disordered[np.where(X_disordered==0)]=-1

	X_data=np.concatenate((X_ordered,X_disordered))
	Y_data=np.concatenate((Y_ordered,Y_disordered))

	del X_ordered, X_disordered



	
	# define test and train data sets
	train_data_X=X_data[:train_size]
	train_data_Y=Y_data[:train_size]

	test_data_X=X_data[train_size:]
	test_data_Y=Y_data[train_size:]

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

	print("finished processing data")

	return Datasets(train=train, validation=validation, test=test)


######################################

class Net(object):
	# build the graph for the model
	def __init__(self,opt_kwargs):

		# define global step for checkpointing
		self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		
		self.n_feats= 1600 #n_feats
		self.n_categories=2
		self.n_samples= 170000 #n_samples

		self.n_hidden_1=100
		self.n_hidden_2=200
		

		self.dropout_keepprob=0.5

		# Step 1: create placeholders for input X and label Y
		self.create_placeholders()
		# Step 2: create weight and bias, initialized to 0 and construct model to predict Y from X
		self.create_model()
		# Step 3: define loss function
		self.create_loss()
		# Step 4: use gradient descent to minimize loss
		self.create_optimiser(opt_kwargs)


		print("finished creating model")

	def create_placeholders(self):
		with tf.name_scope('data'):
			self.X=tf.placeholder(tf.float32, shape=(None,self.n_feats), name="X_data")
			self.Y=tf.placeholder(tf.float32, shape=(None,self.n_categories), name="Y_data")

	def create_model(self):
		with tf.name_scope('model'):

			
			# conv layer 1, 5x5 kernel, 1 input 10 output channels
			W_conv1 = self.weight_variable([5, 5, 1, 10],name='conv1',dtype=tf.float32) 
			b_conv1 = self.bias_variable([10],name='conv1',dtype=tf.float32)
			h_conv1 = tf.nn.relu(self.conv2d(tf.reshape(self.X, [-1, 40, 40, 1]), W_conv1, name='conv1') + b_conv1)

			# Pooling layer - downsamples by 2X.
			h_pool1 = self.max_pool_2x2(h_conv1,name='pool1')

			# conv layer 2, 5x5 kernel, 11 input 20 output channels
			W_conv2 = self.weight_variable([5, 5, 10, 20],name='conv2',dtype=tf.float32)
			b_conv2 = self.bias_variable([20],name='conv2',dtype=tf.float32)
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, name='conv2') + b_conv2)

			# Dropout - controls the complexity of the model, prevents co-adaptation of features.
			h_conv2_drop = tf.nn.dropout(h_conv2, self.dropout_keepprob,name='conv2_dropout')

			# Second pooling layer.
			h_pool2 = self.max_pool_2x2(h_conv2_drop,name='pool2')

			# Fully connected layer 1 -- after 2 round of downsampling, our 40x40 image
			# is down to 7x7x20 feature maps -- maps this to 50 features.
			h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*20])

			W_fc1 = self.weight_variable([7*7*20, 50],name='fc1',dtype=tf.float32)
			b_fc1 = self.bias_variable([50],name='fc1',dtype=tf.float32)

			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

			# Dropout - controls the complexity of the model, prevents co-adaptation of features.
			h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keepprob,name='fc1_dropout')

			# Map the 50 features to 2 classes, one for each phase
			W_fc2 = self.weight_variable([50, self.n_categories],name='fc12',dtype=tf.float32)
			b_fc2 = self.bias_variable([self.n_categories],name='fc12',dtype=tf.float32)

			self.Y_predicted = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	def weight_variable(self, shape, name='', dtype=tf.float64):
		"""weight_variable generates a weight variable of a given shape."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial,dtype=dtype,name=name)


	def bias_variable(self, shape, name='', dtype=tf.float64):
		"""bias_variable generates a bias variable of a given shape."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial,dtype=dtype,name=name)





	def conv2d(self, x, W, name=''):
		"""conv2d returns a 2d convolution layer with full stride."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name=name)


	def max_pool_2x2(self, x,name=''):
		"""max_pool_2x2 downsamples a feature map by 2X."""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], 
								padding='VALID',
								name=name
								)

			


	def create_loss(self):
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(
							tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.Y_predicted)
						)		  
			
	def create_optimiser(self,kwargs):
		with tf.name_scope('optimiser'):
			#self.optimizer = tf.train.GradientDescentOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step) 
			self.optimizer = tf.train.AdamOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)



##################################




def main(_):

	training_epochs=5001
	
	n_samples=100000
	train_size=70000
	validation_size=0
	batch_size=200
	
	# ADAM learning params
	learning_rate=0.001 # learning rate
	beta1=0.94
	beta2=0.89
	epsilon=1e-08
	opt_params=dict(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
	#opt_params=dict(learning_rate=learning_rate)
	
	# Prrocess data
	pkl_file='mag_vs_T_L40_T=3.50.pkl'
	root_dir=os.path.expanduser('~')+'/Dropbox/MachineLearningReview/Datasets/isingMC/compressed/'

	Ising_Data=read_data_sets(pkl_file,root_dir,train_size=train_size,validation_size=validation_size)

	

	# define model
	model=Net(opt_params)
	
	
	with tf.Session() as sess:


		# Step 7: initialize the necessary variables, in this case, w and b
		sess.run(tf.global_variables_initializer())

		# Step 8: train the model
		for epoch in range(training_epochs): 

			batch_X, batch_Y = Ising_Data.train.next_batch(batch_size,seed=seed)
			
			loss_batch, _ = sess.run([model.loss, model.optimizer],
												feed_dict={model.X: batch_X,model.Y: batch_Y   } )
			# count training step
			step = sess.run(model.global_step)

		
			print(epoch ,loss_batch/batch_size, loss_batch)

			
		# Step 9: test model
		train_loss, train_Y, train_Y_predicted = sess.run([model.loss, model.Y, model.Y_predicted], 
													feed_dict={model.X: Ising_Data.train.data_X, 
															   model.Y: Ising_Data.train.data_Y}
															   )
		print("train loss:", train_loss)

		test_loss, test_Y, test_Y_predicted = sess.run([model.loss, model.Y, model.Y_predicted], 
													feed_dict={model.X: Ising_Data.test.data_X,
															   model.Y: Ising_Data.test.data_Y}
															   )
		print("test loss:", test_loss)

				

	
if __name__ == '__main__':

	# define number of samples 
	N_samples=170000
	
	L=40 #spin chain system size
	
	# run ML tool
	tf.app.run(main=main, argv=[sys.argv[0]] )

