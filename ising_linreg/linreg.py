# -*- coding: utf-8 -*-
import sys, os

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

import numpy as np
import scipy.sparse as sp

class Linear_Regression(object):
	# build the graph for the model
	def __init__(self,L,opt_params,symmetric=False,J_const=False,nn=False):

		# define model attributes
		self.symmetric=symmetric # symmetric coupling
		self.J_const=J_const # constant nn interactions
		self.nn=nn # nn interactions 
		
		if self.J_const:
			if not nn:
				print("expectien nn=True!")
				print("exiting..")
				exit()
			self.n_feats=1
			self.symmetric=True
		elif self.nn:
			self.n_feats=4*L**2
		else:
			self.n_feats=L**4

		# system linear system size
		self.L=L


		# define global step for checkpointing
		self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		

		# Step 1: create placeholders for input X and label Y
		self._create_placeholders()
		# Step 2: create weight and bias, initialized to 0 and construct model to predict Y from X
		self._create_model()
		# Step 3: define loss function
		self._create_loss()
		# Step 4: use gradient descent to minimize loss
		self._create_optimiser(opt_params)
		# Step 5: create sumamries
		self._create_summaries()

		self._measure_accuracy()


		


	def _create_placeholders(self):
		with tf.name_scope('data'):
			if self.J_const:
				self.X=tf.placeholder(tf.float32, shape=(None,4*self.L**2), name="X_data")
			else:
				self.X=tf.placeholder(tf.float32, shape=(None,self.n_feats), name="X_data")
			self.Y=tf.placeholder(tf.float32, shape=(None,1), name="Y_data")

	def _create_model(self):
		with tf.name_scope('model'):
			
			if self.J_const:
				# calculate nn couplings
				Jnn=np.zeros((self.L,self.L,self.L,self.L),)
				for i in range(self.L):
					for j in range(self.L):
						for k in [-1,1]:
							for l in [-1,1]:
								Jnn[i,j,(i+k)%self.L,(j+l)%self.L]-=1.0
				Jnn=Jnn.reshape((self.L**4,1))
				Jnn=sp.csr_matrix(Jnn).data
				Jnn=Jnn.reshape(Jnn.shape[0],1)
				# define
				self.Eint=tf.Variable(Jnn,trainable=False,name='J',dtype=tf.float32)
				self.J=tf.Variable( tf.random_normal((self.n_feats,self.n_feats), ),dtype=tf.float32, name="int")
				# compute model
				self.Y_predicted=0.5*self.J*tf.matmul(self.X,self.Eint)

			else:
				# initiate weights
				self.Eint=None
				self.W=tf.Variable( tf.random_normal((self.n_feats,1), ),dtype=tf.float32, name="int")
				if self.symmetric:
					# get indices to do W.T on a flattened W
					inds_T=np.arange(self.n_feats)
					inds_T=inds_T.reshape(int(np.sqrt(self.n_feats)),int(np.sqrt(self.n_feats)))
					inds_T=inds_T.T.reshape(self.n_feats)	
					# symmetrise J
					self.J=0.5*(self.W + tf.gather(self.W, inds_T))
				else:
					self.J=tf.identity(self.W)
				# compute model
				self.Y_predicted=0.5*tf.matmul(self.X,self.J)


			
			
	def _create_loss(self):
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean( tf.nn.l2_loss(self.Y - self.Y_predicted))

	def _create_optimiser(self,kwargs):
		with tf.name_scope('optimiser'):
			#self.optimizer = tf.train.GradientDescentOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)
			self.optimizer = tf.train.AdamOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)

	def _measure_accuracy(self):
		"""to be written"""
		with tf.name_scope('accuracy'):
			pass

	def _create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self.loss)
			#tf.summary.scalar("accuracy", self.accuracy)
			tf.summary.histogram("histogram loss", self.loss)
			# merge all summaries into one op to make it easier to manage
			self.summary_op = tf.summary.merge_all()

